from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from tqdm import tqdm, trange

from nanoforge.config import NanoforgeConfig, load_config
from nanoforge.data.dataset import PackedMemmapDataset, make_torch_batch
from nanoforge.data.tokenizer import load_tokenizer
from nanoforge.generation.sampling import SamplingConfig
from nanoforge.model.transformer import NanoforgeForCausalLM
from nanoforge.profiling import estimate_model_profile
from nanoforge.progress import JsonlMetricLogger
from nanoforge.training.checkpoint import AsyncCheckpointSaver, load_checkpoint, restore_rng_state
from nanoforge.training.health import TrainingHealthMonitor
from nanoforge.training.schedulers import create_scheduler
from nanoforge.training.utils import (
    EMA,
    autocast_dtype,
    configure_named_optimizer,
    ensure_dir,
    grad_global_norm,
    grads_are_finite,
    resolve_device,
    seed_everything,
    set_low_memory_env,
)


class Trainer:
    def __init__(self, config: NanoforgeConfig):
        self.config = config
        self.train_cfg = config.training
        self.model_cfg = config.model
        self.data_cfg = config.data
        if self.train_cfg.low_memory:
            set_low_memory_env()
        seed_everything(self.train_cfg.seed)
        self.device = resolve_device(self.train_cfg.device)
        self.model_cfg.gradient_checkpointing = self.train_cfg.gradient_checkpointing
        self.model = NanoforgeForCausalLM(self.model_cfg).to(self.device)
        if self.train_cfg.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]
        self.optimizer = configure_named_optimizer(
            self.model,
            self.train_cfg.optimizer,
            self.train_cfg.learning_rate,
            self.train_cfg.weight_decay,
            self.train_cfg.betas,
            self.train_cfg.eps,
        )
        self.ema = EMA(self.model, self.train_cfg.ema_decay) if self.train_cfg.ema_decay > 0 else None
        self.train_data = PackedMemmapDataset(self.data_cfg.train_path, self.data_cfg.seq_len)
        self.val_data = PackedMemmapDataset(self.data_cfg.val_path, self.data_cfg.seq_len)
        self.output_dir = ensure_dir(self.train_cfg.output_dir)
        scaler_enabled = self.device.type == "cuda" and autocast_dtype(self.train_cfg.precision, self.device) == torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:
            self.scaler = torch.amp.GradScaler(enabled=scaler_enabled)
        self.writer = self._make_writer()
        self.wandb_run = self._make_wandb()
        self.metric_logger = JsonlMetricLogger(self.output_dir / "metrics.jsonl", reset=True)
        self.checkpoints = AsyncCheckpointSaver(self.train_cfg.async_checkpoint)
        self.health = TrainingHealthMonitor(grad_explosion_factor=self.train_cfg.grad_explosion_factor)
        self.start_step = 0
        if self.train_cfg.resume_from_checkpoint:
            self.start_step = self._resume(self.train_cfg.resume_from_checkpoint)

    def _make_writer(self):
        if not self.train_cfg.tensorboard:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(str(self.output_dir / "tb"))
        except Exception:
            return None

    def _make_wandb(self):
        if not self.train_cfg.wandb:
            return None
        try:
            import wandb

            return wandb.init(project="nanoforge", name=self.train_cfg.run_name, config=self.config)
        except Exception:
            return None

    def _log(self, metrics: dict[str, float], step: int, event: str = "train") -> None:
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self.writer.add_scalar(key, value, step)
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)
        self.metric_logger.log(event, step, metrics)

    def _resume(self, checkpoint_path: str | Path) -> int:
        payload = load_checkpoint(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload.get("rng"))
        step = int(payload.get("step", 0)) + 1
        self._log({"checkpoint/resumed_step": float(step)}, step, event="resume")
        return step

    def _batch(self, split: str, batch_size: int):
        dataset = self.train_data if split == "train" else self.val_data
        return make_torch_batch(dataset, batch_size, str(self.device), self.data_cfg.pin_memory and self.device.type == "cuda")

    def _autocast(self):
        dtype = autocast_dtype(self.train_cfg.precision, self.device)
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype is not None)

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        losses: list[float] = []
        iterator = tqdm(
            range(self.train_cfg.eval_steps),
            desc="eval",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in iterator:
            x, y = self._batch("val", self.train_cfg.micro_batch_size)
            with self._autocast():
                out = self.model(x, labels=y)
            losses.append(float(out.loss.detach().cpu()))
            iterator.set_postfix(loss=f"{losses[-1]:.3f}")
        self.model.train()
        val_loss = sum(losses) / len(losses)
        return val_loss, math.exp(min(20.0, val_loss))

    @torch.no_grad()
    def generate_sample(self, step: int) -> dict[str, float]:
        tokenizer = load_tokenizer(self.data_cfg.tokenizer_type, self.data_cfg.tokenizer_path)
        was_training = self.model.training
        self.model.eval()
        prompt_ids = tokenizer.encode(self.train_cfg.sample_prompt, add_bos=True, add_eos=False)
        ids = torch.tensor([prompt_ids[-self.model_cfg.max_seq_len :]], dtype=torch.long, device=self.device)
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

    def train(self) -> None:
        self.model.train()

        best_val = float("inf")
        last_val = float("nan")
        last_ppl = float("nan")
        stale_evals = 0
        t0 = time.time()
        tokens_per_step = self.train_cfg.micro_batch_size * self.train_cfg.grad_accum_steps * self.data_cfg.seq_len
        total_tokens = tokens_per_step * self.train_cfg.max_steps
        warmup_steps = self.train_cfg.warmup_steps
        if warmup_steps <= 0 and self.train_cfg.warmup_ratio > 0:
            warmup_steps = max(1, int(self.train_cfg.max_steps * self.train_cfg.warmup_ratio))
        scheduler = create_scheduler(
            self.train_cfg.scheduler,
            max_steps=self.train_cfg.max_steps,
            warmup_steps=warmup_steps,
            learning_rate=self.train_cfg.learning_rate,
            min_learning_rate=self.train_cfg.min_learning_rate,
        )
        pbar = trange(
            self.start_step,
            self.train_cfg.max_steps,
            desc=f"train:{self.train_cfg.run_name}",
            unit="step",
            dynamic_ncols=True,
        )
        self._log(
            {
                "run/max_steps": self.train_cfg.max_steps,
                "run/total_tokens": total_tokens,
                "run/parameters": self.model.estimate_num_params(),
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
            0,
            event="start",
        )
        for step in pbar:
            step_t0 = time.time()
            lr = scheduler(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            skip_step = False
            skip_reason = ""
            last_logits = None
            for _ in range(self.train_cfg.grad_accum_steps):
                x, y = self._batch("train", self.train_cfg.micro_batch_size)
                with self._autocast():
                    out = self.model(x, labels=y)
                    loss = out.loss / self.train_cfg.grad_accum_steps
                if not torch.isfinite(loss.detach()):
                    skip_step = True
                    skip_reason = "nonfinite_loss"
                    break
                last_logits = out.logits.detach()
                self.scaler.scale(loss).backward()
                total_loss += float(loss.detach().cpu())  # loss is already the per-token mean
            grad_norm_before = 0.0
            grad_norm_after = 0.0
            if skip_step:
                grad_norm_before = 0.0
                grad_norm_after = 0.0
            elif self.train_cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm_before = grad_global_norm(self.model.parameters())
                if not math.isfinite(grad_norm_before) or not grads_are_finite(self.model.parameters()):
                    skip_step = True
                    skip_reason = "nonfinite_grad"
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
                    grad_norm_after = grad_global_norm(self.model.parameters())
            else:
                grad_norm_before = grad_global_norm(self.model.parameters())
                grad_norm_after = grad_norm_before
                if not math.isfinite(grad_norm_before) or not grads_are_finite(self.model.parameters()):
                    skip_step = True
                    skip_reason = "nonfinite_grad"

            if skip_step:
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.update(new_scale=max(float(self.scaler.get_scale()) * 0.5, 1.0))
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.ema and not skip_step:
                self.ema.update(self.model)

            if step % max(1, self.train_cfg.health_interval) == 0:
                snapshot = self.health.observe(
                    loss=total_loss,
                    grad_norm=grad_norm_before,
                    logits=last_logits,
                    optimizer=self.optimizer,
                    device=self.device,
                )
                self._log(snapshot.metrics, step, event="health")
                for event in snapshot.events:
                    self._log({**event.metrics, "health/event": event.kind}, step, event=event.severity)

            if step % self.train_cfg.log_interval == 0:
                elapsed = max(time.time() - t0, 1e-6)
                toks = (step + 1) * tokens_per_step
                step_time = max(time.time() - step_t0, 1e-9)
                metrics = {
                    "train/loss": total_loss,
                    "train/lr": lr,
                    "train/tokens_per_sec": toks / elapsed,
                    "train/step_time_sec": step_time,
                    "train/grad_norm": grad_norm_after,
                    "train/grad_norm_before_clip": grad_norm_before,
                    "train/grad_norm_after_clip": grad_norm_after,
                    "train/tokens_seen": toks,
                    "train/loss_scale": float(self.scaler.get_scale()) if self.scaler.is_enabled() else 1.0,
                    "train/skipped_step": 1.0 if skip_step else 0.0,
                    "val/loss": last_val,
                    "val/perplexity": last_ppl,
                    "val/best_loss": best_val if best_val < float("inf") else float("nan"),
                }
                if skip_step:
                    self._log({"train/skipped_step": 1.0, "train/skip_reason": skip_reason}, step, event="skip")
                self._log(metrics, step)
                pbar.set_postfix(
                    loss=f"{total_loss:.3f}",
                    val="-" if math.isnan(last_val) else f"{last_val:.3f}",
                    ppl="-" if math.isnan(last_ppl) else f"{last_ppl:.1f}",
                    tok_s=f"{metrics['train/tokens_per_sec']:.0f}",
                    lr=f"{lr:.2e}",
                    gn=f"{grad_norm_before:.1f}->{grad_norm_after:.1f}",
                    skip="yes" if skip_step else "no",
                )

            if step > 0 and step % self.train_cfg.eval_interval == 0:
                val_loss, ppl = self.evaluate()
                last_val, last_ppl = val_loss, ppl
                self._log({"val/loss": val_loss, "val/perplexity": ppl}, step, event="eval")
                if val_loss < best_val:
                    best_val = val_loss
                    stale_evals = 0
                    self._log({"checkpoint/best_val_loss": best_val}, step, event="checkpoint")
                    self.checkpoints.save(
                        self.output_dir / "best.pt",
                        self.model,
                        self.optimizer,
                        self.config,
                        step,
                        val_loss,
                        self.ema.state_dict() if self.ema else None,
                    )
                else:
                    stale_evals += 1
                    if stale_evals >= self.train_cfg.early_stopping_patience:
                        break

            if step > 0 and step % self.train_cfg.save_interval == 0:
                self.checkpoints.save(
                    self.output_dir / f"step-{step}.pt",
                    self.model,
                    self.optimizer,
                    self.config,
                    step,
                    None,
                    self.ema.state_dict() if self.ema else None,
                )

            if self.train_cfg.sample_interval > 0 and step > 0 and step % self.train_cfg.sample_interval == 0:
                self._log(self.generate_sample(step), step, event="sample")

        self.checkpoints.save(
            self.output_dir / "last.pt",
            self.model,
            self.optimizer,
            self.config,
            self.train_cfg.max_steps,
            None,
            self.ema.state_dict() if self.ema else None,
        )
        self.checkpoints.close()
        self._log({"train/final_step": self.train_cfg.max_steps, "val/best_loss": best_val}, self.train_cfg.max_steps, event="done")


def train_from_config(path: str | Path) -> None:
    cfg = load_config(path)
    Trainer(cfg).train()


def _repetition_score(ids: list[int], n: int = 3) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[idx : idx + n]) for idx in range(len(ids) - n + 1)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))


def _distinct_ratio(ids: list[int]) -> float:
    if not ids:
        return 0.0
    return len(set(ids)) / len(ids)
