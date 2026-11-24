from __future__ import annotations

import copy
import inspect
import math
import signal
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from nanoforge.artifacts import RunManifestWriter
from nanoforge.config import NanoforgeConfig, load_config
from nanoforge.data.dataset import PackedMemmapDataset, make_torch_batch
from nanoforge.data.tokenizer import load_tokenizer
from nanoforge.generation.sampling import SamplingConfig
from nanoforge.logging_utils import get_logger
from nanoforge.model.transformer import build_model
from nanoforge.profiling import estimate_model_profile
from nanoforge.progress import JsonlMetricLogger
from nanoforge.training.checkpoint import (
    AsyncCheckpointSaver,
    load_checkpoint,
    make_checkpoint_payload,
    restore_rng_state,
    rng_state,
)
from nanoforge.training.distributed import discover_env, init_process_group, wrap_model
from nanoforge.training.health import ActivationDiagnostics, TrainingHealthMonitor, memory_stats
from nanoforge.training.schedulers import create_scheduler
from nanoforge.training.utils import (
    EMA,
    SWA,
    autocast_dtype,
    configure_named_optimizer,
    ensure_dir,
    grad_global_norm,
    grads_are_finite,
    move_optimizer_state,
    resolve_device,
    seed_everything,
    set_low_memory_env,
)

_SAMPLER_LOCK = threading.RLock()


class _RankLocalManifest:
    """Non-writer ranks participate in computation without creating shared run files."""

    def update(self, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass

    def add_checkpoint(self, *args, **kwargs):
        pass


class Trainer:
    def __init__(self, config: NanoforgeConfig):
        self._owns_process_group = False
        try:
            self._initialize(config)
        except BaseException as exc:
            logger = get_logger("training.trainer")
            for name, method in (("activation_diagnostics", "close"), ("checkpoints", "close"),
                                 ("writer", "close"), ("wandb_run", "finish")):
                resource = getattr(self, name, None)
                if resource:
                    try:
                        getattr(resource, method)()
                    except Exception as cleanup_error:
                        logger.warning("Initialization cleanup also failed: %s", type(cleanup_error).__name__)
            manifest = getattr(self, "run_manifest", None)
            if manifest:
                try:
                    manifest.finish("failed", provenance={"error_type": type(exc).__name__})
                except Exception as cleanup_error:
                    logger.warning("Initialization manifest also failed: %s", type(cleanup_error).__name__)
            if self._owns_process_group:
                import torch.distributed as dist

                if dist.is_initialized():
                    try:
                        dist.destroy_process_group()
                    except Exception as cleanup_error:
                        logger.warning("Process group cleanup also failed: %s", type(cleanup_error).__name__)
            raise

    def _initialize(self, config: NanoforgeConfig):
        self.config = config
        self.train_cfg = config.training
        self.model_cfg = config.model
        self.data_cfg = config.data
        self._stop_requested = False
        self._validate_controls()
        if self.train_cfg.low_memory:
            set_low_memory_env()
        threads = getattr(self.train_cfg, "cpu_threads", 0)
        if threads:
            torch.set_num_threads(threads)
        seed_everything(self.train_cfg.seed)
        self.dist_env = discover_env()
        self._owns_process_group = False
        self.device = resolve_device(self.train_cfg.device)
        if self.dist_env.enabled:
            import torch.distributed as dist

            self._owns_process_group = not dist.is_initialized()
            if self.train_cfg.distributed_backend in {"none", ""}:
                raise ValueError(
                    "WORLD_SIZE > 1 requires an explicit synchronized distributed_backend."
                )
            init_process_group(
                self.dist_env, backend="nccl" if self.device.type == "cuda" else "gloo"
            )
            if self.device.type == "cuda":
                self.device = torch.device("cuda", self.dist_env.local_rank)
        self.is_main_process = self.dist_env.is_main
        self.model_cfg.gradient_checkpointing = self.train_cfg.gradient_checkpointing
        self.model = build_model(config).to(self.device)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        initial = getattr(self.train_cfg, "init_from_checkpoint", None)
        if initial:
            self.model.load_state_dict(
                load_checkpoint(initial, map_location="cpu")["model"], strict=True
            )
        self.model = wrap_model(
            self.model,
            self.dist_env,
            self.train_cfg.distributed_backend,
            precision=self.train_cfg.precision,
            gradient_checkpointing=self.train_cfg.gradient_checkpointing,
        )
        if self.train_cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.optimizer = configure_named_optimizer(
            self.model,
            self.train_cfg.optimizer,
            self.train_cfg.learning_rate,
            self.train_cfg.weight_decay,
            self.train_cfg.betas,
            self.train_cfg.eps,
        )
        if self.train_cfg.distributed_backend == "zero" and self.dist_env.enabled:
            if self.train_cfg.optimizer != "adamw":
                raise ValueError("ZeRO stage 1 currently supports AdamW only.")
            from torch.distributed.optim import ZeroRedundancyOptimizer

            self.optimizer = ZeroRedundancyOptimizer(
                self.optimizer.param_groups,
                optimizer_class=torch.optim.AdamW,
                lr=self.train_cfg.learning_rate,
                betas=self.train_cfg.betas,
                eps=self.train_cfg.eps,
            )
        if self._is_fsdp and (
            self.train_cfg.ema_decay or getattr(self.train_cfg, "swa_start", -1) >= 0
        ):
            raise ValueError(
                "EMA/SWA with sharded FSDP weights is not supported; use DDP or disable averaging."
            )
        self.ema = (
            EMA(self.unwrapped_model, self.train_cfg.ema_decay)
            if self.train_cfg.ema_decay > 0
            else None
        )
        self.swa = (
            SWA(self.unwrapped_model) if getattr(self.train_cfg, "swa_start", -1) >= 0 else None
        )
        rank_seed = self.train_cfg.seed + self.dist_env.rank * 100003
        self._train_rng = np.random.RandomState(rank_seed)
        self._val_rng = np.random.RandomState(self.train_cfg.seed + 7654321)
        self.train_data = self._build_train_data(rank_seed)
        self.val_data = self._dataset(self.data_cfg.val_path, self.train_cfg.seed + 7654321)
        self._validation_state = self._sampler_state(self.val_data)
        seed_everything(rank_seed)
        scaler_enabled = (
            self.device.type == "cuda"
            and autocast_dtype(self.train_cfg.precision, self.device) == torch.float16
        )
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:
            self.scaler = torch.amp.GradScaler(enabled=scaler_enabled)
        self.health = TrainingHealthMonitor(
            grad_explosion_factor=self.train_cfg.grad_explosion_factor
        )
        self.activation_diagnostics = ActivationDiagnostics(self.unwrapped_model)
        self.completed_updates = 0
        self.attempted_updates = 0
        self.skipped_updates = 0
        self.tokens_seen = 0
        self.best_val = float("inf")
        self.stale_evals = 0
        self.last_val = float("nan")
        self.last_ppl = float("nan")
        self.lr_scale = 1.0
        self.start_step = 0
        if self.train_cfg.resume_from_checkpoint:
            self.start_step = self._resume(self.train_cfg.resume_from_checkpoint)
        self.output_dir = ensure_dir(self.train_cfg.output_dir)
        self.run_manifest = (
            RunManifestWriter(
                self.output_dir, self.train_cfg.run_name, self.config, device=str(self.device)
            )
            if self.is_main_process
            else _RankLocalManifest()
        )
        self.run_manifest.update(
            data={
                "train_path": self.data_cfg.train_path,
                "val_path": self.data_cfg.val_path,
                "seq_len": self.data_cfg.seq_len,
                "mode": self.data_cfg.mode,
                "loss_masking": self.data_cfg.loss_masking,
            },
            tokenizer={"type": self.data_cfg.tokenizer_type, "path": self.data_cfg.tokenizer_path},
            model={
                "parameter_count": self.parameter_count,
                "training_mode": "initialized" if initial else "from_scratch",
            },
            provenance={
                "world_size": self.dist_env.world_size,
                "resume_from_checkpoint": self.train_cfg.resume_from_checkpoint,
                "init_from_checkpoint": initial,
            },
        )
        self.writer = self._make_writer()
        self.wandb_run = self._make_wandb()
        self.metric_logger = (
            JsonlMetricLogger(
                self.output_dir / "metrics.jsonl",
                reset=not bool(self.train_cfg.resume_from_checkpoint),
            )
            if self.is_main_process
            else None
        )
        self.checkpoints = AsyncCheckpointSaver(
            self.train_cfg.async_checkpoint and self.is_main_process
        )
        if self.start_step:
            self._log({"checkpoint/resumed_step": self.start_step}, self.start_step, event="resume")

    def _validate_controls(self):
        for name in (
            "micro_batch_size",
            "grad_accum_steps",
            "eval_steps",
            "log_interval",
            "eval_interval",
            "save_interval",
        ):
            if getattr(self.train_cfg, name) <= 0:
                raise ValueError(f"training.{name} must be positive.")
        if self.train_cfg.max_steps < 0:
            raise ValueError("training.max_steps cannot be negative.")
        if (
            getattr(self.train_cfg, "init_from_checkpoint", None)
            and self.train_cfg.resume_from_checkpoint
        ):
            raise ValueError("Weight initialization and full resume are mutually exclusive.")
        if getattr(self.train_cfg, "cpu_threads", 0) < 0:
            raise ValueError("training.cpu_threads cannot be negative.")
        if getattr(self.train_cfg, "swa_interval", 1) <= 0:
            raise ValueError("training.swa_interval must be positive.")
        if self.train_cfg.distributed_backend == "zero" and getattr(
            self.train_cfg, "optimizer_cpu_offload", False
        ):
            raise ValueError("Generic state offload cannot be combined with ZeRO stage 1.")
        if getattr(self.train_cfg, "max_consecutive_skipped_steps", 10) <= 0:
            raise ValueError("training.max_consecutive_skipped_steps must be positive.")
        if not 0 < getattr(self.train_cfg, "recovery_lr_factor", 1.0) <= 1:
            raise ValueError("training.recovery_lr_factor must be in (0, 1].")
        previous = -1
        for stage in getattr(self.train_cfg, "curriculum", []):
            if (
                set(stage) != {"step", "seq_len"}
                or stage["step"] <= previous
                or not 0 < stage["seq_len"] <= self.data_cfg.seq_len
            ):
                raise ValueError(
                    "Curriculum requires ascending nonnegative step/positive seq_len entries within data.seq_len."
                )
            previous = stage["step"]

    @property
    def unwrapped_model(self):
        model = self.model
        while hasattr(model, "module") or hasattr(model, "_orig_mod"):
            model = model.module if hasattr(model, "module") else model._orig_mod
        return model

    @property
    def _is_fsdp(self):
        return self.dist_env.enabled and self.train_cfg.distributed_backend == "fsdp"

    @property
    def _distributed_model(self):
        return self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

    def _dataset(self, path, seed):
        arguments = (
            {"seed": seed} if "seed" in inspect.signature(PackedMemmapDataset).parameters else {}
        )
        return PackedMemmapDataset(path, self.data_cfg.seq_len, **arguments)

    def _build_train_data(self, seed=None):
        from nanoforge.data.mixture import build_mixture_from_config

        mixture = build_mixture_from_config(
            self.data_cfg, self.data_cfg.seq_len, seed or self.train_cfg.seed
        )
        return (
            mixture
            if mixture is not None
            else self._dataset(self.data_cfg.train_path, seed or self.train_cfg.seed)
        )

    @staticmethod
    def _sampler_state(dataset):
        state = {}
        if hasattr(dataset, "state_dict"):
            state["dataset"] = copy.deepcopy(dataset.state_dict())
        if hasattr(dataset, "_rng") and hasattr(dataset._rng, "bit_generator"):
            state["mixture"] = copy.deepcopy(dataset._rng.bit_generator.state)
        return state

    @staticmethod
    def _restore_sampler(dataset, state):
        if "dataset" in state and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(state["dataset"])
        if "mixture" in state and hasattr(dataset, "_rng"):
            dataset._rng.bit_generator.state = state["mixture"]

    def _make_writer(self):
        if not self.is_main_process or not self.train_cfg.tensorboard:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(str(self.output_dir / "tb"))
        except (ImportError, OSError) as exc:
            get_logger("training.trainer").warning("TensorBoard unavailable: %s", exc)
            return None

    def _make_wandb(self):
        if not self.is_main_process or not self.train_cfg.wandb:
            return None
        import wandb

        return wandb.init(project="nanoforge", name=self.train_cfg.run_name, config=self.config)

    def _log(self, metrics: dict[str, Any], step: int, event: str = "train"):
        if not self.is_main_process:
            return
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self.writer.add_scalar(key, value, step)
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)
        self.metric_logger.log(event, step, metrics)

    def _resume(self, path):
        payload = load_checkpoint(path, map_location="cpu")
        if self._is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            with FSDP.state_dict_type(self._distributed_model, StateDictType.FULL_STATE_DICT):
                self._distributed_model.load_state_dict(payload["model"], strict=True)
                if payload.get("optimizer"):
                    self.optimizer.load_state_dict(
                        FSDP.optim_state_dict_to_load(
                            self._distributed_model, self.optimizer, payload["optimizer"]
                        )
                    )
        else:
            self.unwrapped_model.load_state_dict(payload["model"], strict=True)
            if payload.get("optimizer") is not None:
                self.optimizer.load_state_dict(payload["optimizer"])
        state = payload.get("training_state", {})
        self.completed_updates = int(payload["next_step"])
        for key in (
            "attempted_updates",
            "skipped_updates",
            "tokens_seen",
            "best_val",
            "stale_evals",
            "last_val",
            "last_ppl",
            "lr_scale",
        ):
            if key in state:
                setattr(self, key, state[key])
        if self.ema:
            if payload.get("ema") is not None:
                self.ema.load_state_dict(payload["ema"])
            else:
                self.ema = EMA(self.unwrapped_model, self.train_cfg.ema_decay)
        if self.swa and state.get("swa") is not None:
            self.swa.load_state_dict(state["swa"])
            self.swa.updates = state.get("swa_updates", 0)
        if state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        if state.get("health"):
            self.health.load_state_dict(state["health"])
        rank_states = state.get("rank_states", {})
        if rank_states and len(rank_states) != self.dist_env.world_size:
            raise ValueError(
                "Exact resume requires the checkpoint world size; initialize weights for a new topology."
            )
        rank_state = rank_states.get(self.dist_env.rank, {})
        restore_rng_state(rank_state.get("rng", payload.get("rng")))
        if rank_state:
            self._train_rng.set_state(rank_state["numpy_sampler"])
            self._restore_sampler(self.train_data, rank_state.get("sampler", {}))
        if getattr(self.train_cfg, "optimizer_cpu_offload", False):
            move_optimizer_state(self.optimizer, to_cpu=True)
        return self.completed_updates

    def _batch(self, split: str, batch_size: int):
        dataset = self.train_data if split == "train" else self.val_data
        generator = self._train_rng if split == "train" else self._val_rng
        with _SAMPLER_LOCK:
            previous = np.random.get_state()
            np.random.set_state(generator.get_state())
            try:
                batch = make_torch_batch(
                    dataset,
                    batch_size,
                    str(self.device),
                    self.data_cfg.pin_memory and self.device.type == "cuda",
                )
                generator.set_state(np.random.get_state())
            finally:
                np.random.set_state(previous)
        if split == "train":
            length = self.data_cfg.seq_len
            for stage in getattr(self.train_cfg, "curriculum", []):
                if self.completed_updates >= stage["step"]:
                    length = stage["seq_len"]
            return batch[0][:, :length], batch[1][:, :length] if batch[1].ndim > 1 else batch[1]
        return batch

    def _autocast(self):
        dtype = autocast_dtype(self.train_cfg.precision, self.device)
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype is not None)

    def _forward_loss(self, x, y):
        output = self.model(x, labels=y)
        return output.loss, output.logits

    def _any(self, condition: bool) -> bool:
        if not self.dist_env.enabled:
            return condition
        import torch.distributed as dist

        value = torch.tensor(int(condition), device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return bool(value.item())

    def _sum(self, value):
        if self.dist_env.enabled:
            import torch.distributed as dist

            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    @torch.no_grad()
    def evaluate(self):
        was_training = self.model.training
        model_rng = rng_state()
        sampler = self._sampler_state(self.val_data)
        generator = self._val_rng.get_state()
        self._restore_sampler(self.val_data, self._validation_state)
        self._val_rng.seed(self.train_cfg.seed + 7654321)
        average = (
            self.swa
            if getattr(self.train_cfg, "swa_eval", False) and self.swa and self.swa.updates
            else (self.ema if getattr(self.train_cfg, "ema_eval", False) and self.ema else None)
        )
        total = torch.zeros(2, dtype=torch.float64, device=self.device)
        try:
            self.model.eval()
            with average.applied(self.unwrapped_model) if average else nullcontext():
                for _ in range(self.train_cfg.eval_steps):
                    x, y = self._batch("val", self.train_cfg.micro_batch_size)
                    with self._autocast():
                        loss, _ = self._forward_loss(x, y)
                    count = int((y != -100).sum())
                    total += torch.tensor(
                        [float(loss) * count, count], dtype=torch.float64, device=self.device
                    )
            self._sum(total)
            if total[1] <= 0 or not torch.isfinite(total[0]):
                raise ValueError("Validation has no valid labels or a nonfinite loss.")
            val_loss = float(total[0] / total[1])
            return val_loss, math.exp(min(20.0, val_loss))
        finally:
            self.model.train(was_training)
            restore_rng_state(model_rng)
            self._val_rng.set_state(generator)
            self._restore_sampler(self.val_data, sampler)

    @torch.no_grad()
    def generate_sample(self, step: int) -> dict[str, float]:
        if not getattr(self.model_cfg, "is_causal", True):
            return {}  # sampling only meaningful for generative tasks
        tokenizer = load_tokenizer(self.data_cfg.tokenizer_type, self.data_cfg.tokenizer_path)
        was_training = self.model.training
        self.model.eval()
        prompt_ids = tokenizer.encode(self.train_cfg.sample_prompt, add_bos=True, add_eos=False)
        ids = torch.tensor(
            [prompt_ids[-self.model_cfg.max_seq_len :]], dtype=torch.long, device=self.device
        )
        generated: list[int] = []
        sampling = SamplingConfig(
            mode=self.config.inference.mode,
            temperature=self.config.inference.temperature,
            top_k=self.config.inference.top_k,
            top_p=self.config.inference.top_p,
            min_p=self.config.inference.min_p,
            repetition_penalty=self.config.inference.repetition_penalty,
            frequency_penalty=self.config.inference.frequency_penalty,
            presence_penalty=self.config.inference.presence_penalty,
            no_repeat_ngram_size=self.config.inference.no_repeat_ngram_size,
            deterministic=self.config.inference.deterministic,
            stop_on_repetition=self.config.inference.stop_on_repetition,
            repetition_window=self.config.inference.repetition_window,
            repetition_threshold=self.config.inference.repetition_threshold,
            mirostat=self.config.inference.mirostat,
            mirostat_version=self.config.inference.mirostat_version,
            mirostat_tau=self.config.inference.mirostat_tau,
            mirostat_eta=self.config.inference.mirostat_eta,
        )
        from nanoforge.generation.sampling import MirostatState, sample_next

        caches = None
        miro = MirostatState(sampling.mirostat_tau) if sampling.mirostat else None
        for token_step in range(self.train_cfg.sample_max_new_tokens):
            with self._autocast():
                if token_step == 0:
                    out = self.model(ids[:, -self.model_cfg.max_seq_len :], use_cache=True)
                else:
                    out = self.model(ids[:, -1:], caches=caches, use_cache=True)
            caches = out.caches
            token = sample_next(out.logits, ids[0], sampling, miro)
            next_id = int(token.item())
            if next_id == tokenizer.eos_id:
                break
            generated.append(next_id)
            ids = torch.cat([ids, token], dim=1)
        text = tokenizer.decode(generated)
        sample_dir = self.output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = sample_dir / f"sample_{step}.txt"
        path.write_text(self.train_cfg.sample_prompt + text, encoding="utf-8")
        if was_training:
            self.model.train()
        repetition = _repetition_score(generated)
        diversity = _distinct_ratio(generated)
        return {
            "sample/repetition": repetition,
            "sample/distinct_token_ratio": diversity,
            "sample/tokens": float(len(generated)),
        }

    def request_stop(self):
        """Cooperative cancellation is synchronized at the next update boundary."""
        self._stop_requested = True

    def _training_state(self):
        local = {
            "rng": rng_state(),
            "numpy_sampler": self._train_rng.get_state(),
            "sampler": self._sampler_state(self.train_data),
        }
        ranks = [local]
        if self.dist_env.enabled:
            import torch.distributed as dist

            ranks = [None] * self.dist_env.world_size
            dist.all_gather_object(ranks, local)
        return {
            "attempted_updates": self.attempted_updates,
            "skipped_updates": self.skipped_updates,
            "tokens_seen": self.tokens_seen,
            "best_val": self.best_val,
            "stale_evals": self.stale_evals,
            "last_val": self.last_val,
            "last_ppl": self.last_ppl,
            "lr_scale": self.lr_scale,
            "scaler": self.scaler.state_dict(),
            "health": self.health.state_dict(),
            "swa": self.swa.state_dict() if self.swa else None,
            "swa_updates": self.swa.updates if self.swa else 0,
            "rank_states": dict(enumerate(ranks)),
        }

    def _save(self, name, val_loss=None):
        state = self._training_state()
        failure = None
        try:
            if self.train_cfg.distributed_backend == "zero" and self.dist_env.enabled:
                self.optimizer.consolidate_state_dict(to=0)
            if self._is_fsdp:
                from torch.distributed.fsdp import (
                    FullOptimStateDictConfig,
                    FullStateDictConfig,
                    StateDictType,
                )
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                )

                with FSDP.state_dict_type(
                    self._distributed_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                    FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
                ):
                    payload = make_checkpoint_payload(
                        self._distributed_model,
                        None,
                        self.config,
                        self.completed_updates,
                        val_loss,
                        training_state=state,
                    )
                    optimizer_state = FSDP.optim_state_dict(self._distributed_model, self.optimizer)
                if self.is_main_process:
                    payload["optimizer"] = optimizer_state
                    payload["parameter_count"] = self.parameter_count
                    self.checkpoints.save_payload(self.output_dir / name, payload)
            elif self.is_main_process:
                self.checkpoints.save(
                    self.output_dir / name,
                    self.unwrapped_model,
                    self.optimizer,
                    self.config,
                    self.completed_updates,
                    val_loss,
                    self.ema.state_dict() if self.ema else None,
                    training_state=state,
                )
        except Exception as exc:
            failure = exc
        if self._any(failure is not None):
            if failure:
                raise failure
            raise RuntimeError("A distributed rank could not save the checkpoint.")

    def _train_update(self):
        self.activation_diagnostics.begin(
            self.completed_updates % max(1, self.train_cfg.health_interval) == 0
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        valid_count = 0
        tokens = 0
        last_logits = None
        skip_reason = ""
        for microstep in range(self.train_cfg.grad_accum_steps):
            x, y = self._batch("train", self.train_cfg.micro_batch_size)
            count = int((y != -100).sum())
            tokens += x.numel()
            synchronize = microstep == self.train_cfg.grad_accum_steps - 1
            context = (
                self.model.no_sync()
                if hasattr(self.model, "no_sync") and not synchronize
                else nullcontext()
            )
            with context:
                with self._autocast():
                    mean_loss, logits = self._forward_loss(x, y)
                    summed_loss = mean_loss * count
                if self._any(not bool(torch.isfinite(summed_loss.detach()))):
                    skip_reason = "nonfinite_loss"
                    break
                self.scaler.scale(summed_loss).backward()
                loss_sum += float(summed_loss.detach())
                valid_count += count
                last_logits = logits.detach()
        totals = self._sum(
            torch.tensor([loss_sum, valid_count, tokens], dtype=torch.float64, device=self.device)
        )
        if not skip_reason and totals[1] == 0:
            skip_reason = "empty_labels"
        before = after = 0.0
        if not skip_reason:
            self.scaler.unscale_(self.optimizer)
            multiplier = self.dist_env.world_size / float(totals[1])
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(multiplier)
            if self._any(not grads_are_finite(self.model.parameters())):
                skip_reason = "nonfinite_grad"
            else:
                if self._is_fsdp:
                    before = float(
                        self._distributed_model.clip_grad_norm_(
                            self.train_cfg.grad_clip
                            if self.train_cfg.grad_clip > 0
                            else float("inf")
                        )
                    )
                    after = (
                        min(before, self.train_cfg.grad_clip)
                        if self.train_cfg.grad_clip > 0
                        else before
                    )
                else:
                    before = grad_global_norm(self.model.parameters())
                    if self.train_cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.train_cfg.grad_clip
                        )
                    after = grad_global_norm(self.model.parameters())
                if self._any(not math.isfinite(before)):
                    skip_reason = "nonfinite_grad_norm"
        if skip_reason:
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.update(new_scale=max(float(self.scaler.get_scale()) * 0.5, 1.0))
        else:
            if getattr(self.train_cfg, "optimizer_cpu_offload", False):
                move_optimizer_state(self.optimizer, to_cpu=False)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if getattr(self.train_cfg, "optimizer_cpu_offload", False):
                move_optimizer_state(self.optimizer, to_cpu=True)
        activation_metrics = self.activation_diagnostics.summary()
        self.activation_diagnostics.enabled = False
        return {
            "loss": float(totals[0] / totals[1]) if totals[1] else float("nan"),
            "tokens": int(totals[2]),
            "valid_tokens": int(totals[1]),
            "grad_norm_before": before,
            "grad_norm_after": after,
            "skip_reason": skip_reason,
            "logits": last_logits,
            "activation_metrics": activation_metrics,
        }

    def train(self):
        """Run update-budgeted training; every exit restores handlers and finalizes writers."""
        logger = get_logger("training.trainer")
        previous_handlers = []

        def stop(signum, frame):
            if self._stop_requested:
                raise KeyboardInterrupt
            self.request_stop()

        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers.append((sig, signal.getsignal(sig)))
                signal.signal(sig, stop)
        started = time.perf_counter()
        status = "failed"
        reason = "error"
        failure = None
        progress = None
        try:
            self.model.train()
            warmup = self.train_cfg.warmup_steps
            if warmup <= 0 and self.train_cfg.warmup_ratio > 0:
                warmup = max(1, int(self.train_cfg.max_steps * self.train_cfg.warmup_ratio))
            schedule = create_scheduler(
                self.train_cfg.scheduler,
                max_steps=self.train_cfg.max_steps,
                warmup_steps=warmup,
                learning_rate=self.train_cfg.learning_rate,
                min_learning_rate=self.train_cfg.min_learning_rate,
            )
            session_tokens = 0
            consecutive_skips = 0
            self._log(
                {
                    "run/max_steps": self.train_cfg.max_steps,
                    "run/parameters": self.parameter_count,
                    "run/world_size": self.dist_env.world_size,
                    "run/effective_batch_size": self.train_cfg.micro_batch_size
                    * self.train_cfg.grad_accum_steps
                    * self.dist_env.world_size,
                    **{
                        f"profile/{key}": value
                        for key, value in estimate_model_profile(
                            self.model_cfg,
                            batch_size=self.train_cfg.micro_batch_size,
                            seq_len=self.data_cfg.seq_len,
                        )
                        .to_dict()
                        .items()
                    },
                },
                self.completed_updates,
                event="start",
            )
            progress = tqdm(
                total=self.train_cfg.max_steps,
                initial=self.completed_updates,
                desc=f"train:{self.train_cfg.run_name}",
                unit="update",
                disable=not self.is_main_process,
            )
            reason = "max_steps"
            while self.completed_updates < self.train_cfg.max_steps:
                if self._any(self._stop_requested):
                    status, reason = "interrupted", "cancelled"
                    self._save("interrupt.pt")
                    break
                step_started = time.perf_counter()
                lr = schedule(self.completed_updates) * self.lr_scale
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                result = self._train_update()
                self.attempted_updates += 1
                session_tokens += result["tokens"]
                self.tokens_seen += result["tokens"]
                if result["skip_reason"]:
                    self.skipped_updates += 1
                    consecutive_skips += 1
                    self.lr_scale *= getattr(self.train_cfg, "recovery_lr_factor", 1.0)
                    self._log(
                        {
                            "train/skip_reason": result["skip_reason"],
                            "health/nonfinite_activation_modules": list(self.activation_diagnostics.nonfinite_modules),
                            "train/skipped_step": 1,
                            "train/attempted_updates": self.attempted_updates,
                        },
                        self.completed_updates,
                        event="skip",
                    )
                    if consecutive_skips >= getattr(
                        self.train_cfg, "max_consecutive_skipped_steps", 10
                    ):
                        raise RuntimeError(
                            "Repeated nonfinite or empty-label updates exceeded the recovery budget."
                        )
                    continue
                consecutive_skips = 0
                self.completed_updates += 1
                if self.ema:
                    self.ema.update(self.unwrapped_model)
                if (
                    self.swa
                    and self.completed_updates >= getattr(self.train_cfg, "swa_start", -1)
                    and (self.completed_updates - getattr(self.train_cfg, "swa_start", -1))
                    % getattr(self.train_cfg, "swa_interval", 1)
                    == 0
                ):
                    self.swa.update(self.unwrapped_model)
                progress.update(1)
                if (self.completed_updates - 1) % max(1, self.train_cfg.health_interval) == 0:
                    snapshot = self.health.observe(
                        loss=result["loss"],
                        grad_norm=result["grad_norm_before"],
                        logits=result["logits"],
                        optimizer=getattr(self.optimizer, "optim", self.optimizer),
                        device=self.device,
                        activation_metrics=result["activation_metrics"],
                    )
                    self._log(snapshot.metrics, self.completed_updates, event="health")
                    routing = [
                        module.routing_stats
                        for module in self.unwrapped_model.modules()
                        if hasattr(module, "routing_stats") and module.routing_stats
                    ]
                    if routing:
                        self._log(
                            {
                                "health/moe_assignments": sum(
                                    item.get("assignments", 0) for item in routing
                                ),
                                "health/moe_dropped_assignments": sum(
                                    item.get("dropped_assignments", 0) for item in routing
                                ),
                            },
                            self.completed_updates,
                            event="health",
                        )
                    for event in snapshot.events:
                        self._log(
                            {**event.metrics, "health/event": event.kind},
                            self.completed_updates,
                            event=event.severity,
                        )
                if (self.completed_updates - 1) % self.train_cfg.log_interval == 0:
                    metrics = {
                        "train/loss": result["loss"],
                        "train/lr": lr,
                        "train/tokens_per_sec": session_tokens
                        / max(time.perf_counter() - started, 1e-9),
                        "train/step_time_sec": time.perf_counter() - step_started,
                        "train/grad_norm": result["grad_norm_after"],
                        "train/grad_norm_before_clip": result["grad_norm_before"],
                        "train/grad_norm_after_clip": result["grad_norm_after"],
                        "train/tokens_seen": self.tokens_seen,
                        "train/session_tokens": session_tokens,
                        "train/valid_tokens": result["valid_tokens"],
                        "train/skipped_step": 0,
                        "train/loss_scale": float(self.scaler.get_scale())
                        if self.scaler.is_enabled()
                        else 1.0,
                        "val/loss": self.last_val,
                        "val/perplexity": self.last_ppl,
                        "val/best_loss": self.best_val,
                    }
                    self._log(metrics, self.completed_updates)
                    progress.set_postfix(loss=f"{result['loss']:.3f}", lr=f"{lr:.2e}")
                if self.completed_updates % self.train_cfg.eval_interval == 0:
                    self.last_val, self.last_ppl = self.evaluate()
                    self._log(
                        {"val/loss": self.last_val, "val/perplexity": self.last_ppl},
                        self.completed_updates,
                        event="eval",
                    )
                    if self.last_val < self.best_val:
                        self.best_val, self.stale_evals = self.last_val, 0
                        self._save("best.pt", self.last_val)
                    else:
                        self.stale_evals += 1
                        if self.stale_evals >= self.train_cfg.early_stopping_patience:
                            reason = "early_stopped"
                            break
                if (
                    self.train_cfg.sample_interval > 0
                    and self.completed_updates % self.train_cfg.sample_interval == 0
                ):
                    # Generation monitoring must not consume the optimization RNG stream.
                    state = rng_state()
                    try:
                        if not self.dist_env.enabled:
                            self._log(
                                self.generate_sample(self.completed_updates),
                                self.completed_updates,
                                event="sample",
                            )
                    finally:
                        restore_rng_state(state)
                if self.completed_updates % self.train_cfg.save_interval == 0:
                    self._save(f"step-{self.completed_updates}.pt")
            if status != "interrupted":
                status = "completed"
                self._save("last.pt")
        except BaseException as exc:
            failure = exc
            status = "failed"
            reason = "error"
        finally:
            if progress is not None:
                progress.close()
            try:
                self.checkpoints.close()
            except Exception as exc:
                if failure is None:
                    failure, status = exc, "failed"
                else:
                    logger.warning("Checkpoint finalization also failed: %s", type(exc).__name__)
            try:
                if self.is_main_process:
                    for path in sorted(self.output_dir.glob("*.pt")):
                        self.run_manifest.add_checkpoint(path)
                    self.run_manifest.finish(
                        status,
                        resources={
                            "final_step": self.completed_updates,
                            "best_val_loss": self.best_val,
                            "termination_reason": reason,
                            "attempted_updates": self.attempted_updates,
                            "skipped_updates": self.skipped_updates,
                            "tokens_seen": self.tokens_seen,
                            "elapsed_seconds": time.perf_counter() - started,
                            **memory_stats(self.device),
                        },
                        provenance={"error_type": type(failure).__name__} if failure else {},
                    )
                    self._log(
                        {
                            "train/final_step": self.completed_updates,
                            "val/best_loss": self.best_val,
                            "run/termination_reason": reason,
                        },
                        self.completed_updates,
                        event="interrupted"
                        if status == "interrupted"
                        else ("failed" if failure else "done"),
                    )
            except Exception as exc:
                if failure is None:
                    failure = exc
                else:
                    logger.warning("Run manifest finalization also failed: %s", type(exc).__name__)
            finally:
                self.activation_diagnostics.close()
                for sig, handler in previous_handlers:
                    signal.signal(sig, handler)
                if self.writer:
                    try:
                        self.writer.close()
                    except Exception as exc:
                        failure = failure or exc
                if self.wandb_run:
                    try:
                        self.wandb_run.finish()
                    except Exception as exc:
                        failure = failure or exc
                if self._owns_process_group:
                    import torch.distributed as dist

                    dist.destroy_process_group()
        if failure:
            raise failure


def train_from_config(path: str | Path):
    Trainer(load_config(path)).train()


def run_training_with_optional_tui(path: str | Path, no_tui: bool = False):
    import os
    import sys

    disabled = no_tui or os.environ.get("NANOFORGE_NO_TUI", "") not in ("", "0", "false", "False")
    if disabled or not bool(getattr(sys.stdout, "isatty", lambda: False)()):
        train_from_config(path)
        return
    try:
        from nanoforge.tui import launch_training_tui
    except ImportError:
        train_from_config(path)
        return
    trainer = Trainer(load_config(path))
    state = {"error": None, "done": False}

    def run():
        try:
            trainer.train()
        except BaseException as exc:
            state["error"] = exc
        finally:
            state["done"] = True

    worker = threading.Thread(target=run, name="nanoforge-train")
    worker.start()
    try:
        launch_training_tui(
            str(trainer.output_dir),
            stop_callback=trainer.request_stop,
            done_flag=lambda: state["done"],
        )
    finally:
        try:
            worker.join()
        except KeyboardInterrupt:
            trainer.request_stop()
            worker.join()
    if state["error"]:
        raise state["error"]


def _repetition_score(ids: list[int], n: int = 3):
    grams = [tuple(ids[index : index + n]) for index in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def _distinct_ratio(ids: list[int]):
    return len(set(ids)) / len(ids) if ids else 0.0
