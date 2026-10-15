from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from tqdm import trange

from nanoforge.artifacts import RunManifestWriter
from nanoforge.config import NanoforgeConfig, load_config
from nanoforge.data.tokenizer import load_tokenizer
from nanoforge.logging_utils import get_logger
from nanoforge.model.transformer import build_model
from nanoforge.progress import JsonlMetricLogger
from nanoforge.training.checkpoint import AsyncCheckpointSaver, load_checkpoint
from nanoforge.training.preferences import dpo_loss, sequence_logprob
from nanoforge.training.schedulers import create_scheduler
from nanoforge.training.utils import (
    autocast_dtype,
    configure_named_optimizer,
    ensure_dir,
    grads_are_finite,
    resolve_device,
    seed_everything,
)


def _iter_preference_jsonl(path: str | Path, tokenizer, seq_len: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (chosen_ids, rejected_ids) tensors from a JSONL preference file.

    Each line must have ``chosen`` and ``rejected`` string fields.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            chosen = record.get("chosen", "")
            rejected = record.get("rejected", "")
            if not chosen or not rejected:
                continue
            c_ids = tokenizer.encode(chosen, add_bos=True, add_eos=True)[:seq_len]
            r_ids = tokenizer.encode(rejected, add_bos=True, add_eos=True)[:seq_len]
            yield (
                torch.tensor(c_ids, dtype=torch.long),
                torch.tensor(r_ids, dtype=torch.long),
            )


def _pad_batch(seqs: list[torch.Tensor], pad_id: int = 0) -> torch.Tensor:
    max_len = max(s.shape[0] for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out


def _batch_logprob(model: nn.Module, ids: torch.Tensor, autocast_ctx) -> torch.Tensor:
    """Compute per-sequence log-prob under ``model`` for a batch of token id tensors."""
    with autocast_ctx:
        out = model(ids)
    return sequence_logprob(out.logits, ids)


class DPOTrainer:
    """Direct Preference Optimization trainer.

    Trains a policy model against a frozen reference model using DPO loss.
    The reference model is initialised from the same checkpoint as the policy
    (or from ``reference_checkpoint`` if provided).

    Args:
        config: NanoforgeConfig with training/data/model settings.
        preference_data: Path to a JSONL file with ``chosen`` / ``rejected`` fields.
        beta: KL-regularisation coefficient (default 0.1).
        reference_checkpoint: Optional path to a separate reference checkpoint.
            If None, the reference is a frozen copy of the initial policy weights.
    """

    def __init__(
        self,
        config: NanoforgeConfig,
        preference_data: str | Path,
        beta: float = 0.1,
        reference_checkpoint: str | Path | None = None,
    ):
        self.config = config
        self.train_cfg = config.training
        self.data_cfg = config.data
        self.beta = beta
        self.preference_data = Path(preference_data)

        seed_everything(self.train_cfg.seed)
        self.device = resolve_device(self.train_cfg.device)
        self.logger = get_logger("training.dpo")

        # Policy model
        self.model = build_model(config).to(self.device)
        if self.train_cfg.resume_from_checkpoint:
            payload = load_checkpoint(self.train_cfg.resume_from_checkpoint, map_location=self.device)
            self.model.load_state_dict(payload["model"], strict=True)

        # Reference model — frozen copy
        if reference_checkpoint:
            self.ref_model = build_model(config).to(self.device)
            payload = load_checkpoint(reference_checkpoint, map_location=self.device)
            self.ref_model.load_state_dict(payload["model"], strict=True)
        else:
            self.ref_model = copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        self.tokenizer = load_tokenizer(self.data_cfg.tokenizer_type, self.data_cfg.tokenizer_path)
        self.optimizer = configure_named_optimizer(
            self.model,
            self.train_cfg.optimizer,
            self.train_cfg.learning_rate,
            self.train_cfg.weight_decay,
            self.train_cfg.betas,
            self.train_cfg.eps,
        )
        self.output_dir = ensure_dir(self.train_cfg.output_dir)
        self.run_manifest = RunManifestWriter(
            self.output_dir,
            self.train_cfg.run_name,
            self.config,
            device=str(self.device),
        )
        self.run_manifest.update(
            data={"preference_data": str(self.preference_data), "seq_len": self.data_cfg.seq_len},
            model={
                "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
                "training_mode": "dpo",
            },
            provenance={"reference_checkpoint": str(reference_checkpoint) if reference_checkpoint else None},
        )
        self.metric_logger = JsonlMetricLogger(self.output_dir / "metrics.jsonl", reset=True)
        self.checkpoints = AsyncCheckpointSaver(self.train_cfg.async_checkpoint)

        scaler_enabled = self.device.type == "cuda" and autocast_dtype(self.train_cfg.precision, self.device) == torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except TypeError:
            self.scaler = torch.amp.GradScaler(enabled=scaler_enabled)

    def _autocast(self):
        dtype = autocast_dtype(self.train_cfg.precision, self.device)
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=dtype is not None)

    def _log(self, metrics: dict, step: int, event: str = "train") -> None:
        self.metric_logger.log(event, step, metrics)

    def train(self) -> None:
        self.model.train()
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

        # Pre-load all preference pairs (DPO datasets are typically small)
        pairs = list(_iter_preference_jsonl(self.preference_data, self.tokenizer, self.data_cfg.seq_len))
        if not pairs:
            raise ValueError(f"No valid preference pairs found in {self.preference_data}")
        self.logger.info("DPO: loaded %d preference pairs, beta=%.3f", len(pairs), self.beta)

        t0 = time.time()
        pbar = trange(self.train_cfg.max_steps, desc=f"dpo:{self.train_cfg.run_name}", unit="step", dynamic_ncols=True)
        best_loss = float("inf")

        for step in pbar:
            lr = scheduler(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)

            # Sample a micro-batch of pairs
            indices = torch.randint(len(pairs), (self.train_cfg.micro_batch_size,))
            chosen_seqs = [pairs[i][0] for i in indices]
            rejected_seqs = [pairs[i][1] for i in indices]
            chosen_ids = _pad_batch(chosen_seqs).to(self.device)
            rejected_ids = _pad_batch(rejected_seqs).to(self.device)

            with self._autocast():
                policy_chosen_logp = _batch_logprob(self.model, chosen_ids, self._autocast())
                policy_rejected_logp = _batch_logprob(self.model, rejected_ids, self._autocast())
            with torch.no_grad():
                with self._autocast():
                    ref_chosen_logp = _batch_logprob(self.ref_model, chosen_ids, self._autocast())
                    ref_rejected_logp = _batch_logprob(self.ref_model, rejected_ids, self._autocast())

            loss = dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, self.beta)

            if not torch.isfinite(loss):
                self.logger.warning("step %d: non-finite DPO loss, skipping", step)
                continue

            self.scaler.scale(loss).backward()
            if self.train_cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                if grads_are_finite(self.model.parameters()):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = float(loss.detach().cpu())
            if step % self.train_cfg.log_interval == 0:
                elapsed = max(time.time() - t0, 1e-6)
                metrics = {"dpo/loss": loss_val, "dpo/lr": lr, "dpo/elapsed": elapsed}
                self._log(metrics, step)
                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

            if loss_val < best_loss:
                best_loss = loss_val
                self.checkpoints.save(
                    self.output_dir / "best.pt", self.model, self.optimizer,
                    self.config, step, loss_val, None,
                )

            if step > 0 and step % self.train_cfg.save_interval == 0:
                self.checkpoints.save(
                    self.output_dir / f"step-{step}.pt", self.model, self.optimizer,
                    self.config, step, None, None,
                )

        self.checkpoints.save(
            self.output_dir / "last.pt", self.model, self.optimizer,
            self.config, self.train_cfg.max_steps, None, None,
        )
        self.checkpoints.close()
        for checkpoint_path in sorted(self.output_dir.glob("*.pt")):
            self.run_manifest.add_checkpoint(checkpoint_path)
        self.run_manifest.finish("completed", resources={"final_step": self.train_cfg.max_steps})
        self._log({"dpo/best_loss": best_loss}, self.train_cfg.max_steps, event="done")


def dpo_from_config(config_path: str | Path, preference_data: str | Path, beta: float = 0.1) -> None:
    cfg = load_config(config_path)
    trainer = DPOTrainer(cfg, preference_data, beta=beta)
    try:
        trainer.train()
    except Exception as exc:
        trainer.run_manifest.finish("failed", provenance={"error": f"{type(exc).__name__}: {exc}"})
        raise
