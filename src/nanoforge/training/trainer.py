from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from tqdm import tqdm, trange

from nanoforge.config import NanoforgeConfig, load_config
from nanoforge.data.dataset import PackedMemmapDataset, make_torch_batch
from nanoforge.model.transformer import NanoforgeForCausalLM
from nanoforge.progress import JsonlMetricLogger
from nanoforge.training.checkpoint import save_checkpoint
from nanoforge.training.utils import (
    EMA,
    autocast_dtype,
    configure_named_optimizer,
    cosine_lr,
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
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.device.type == "cuda" and autocast_dtype(self.train_cfg.precision, self.device) == torch.float16
        )
        self.writer = self._make_writer()
        self.wandb_run = self._make_wandb()
        self.metric_logger = JsonlMetricLogger(self.output_dir / "metrics.jsonl", reset=True)

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
        pbar = trange(
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
            },
            0,
            event="start",
        )
        for step in pbar:
            step_t0 = time.time()
            lr = cosine_lr(
                step,
                self.train_cfg.max_steps,
                warmup_steps,
                self.train_cfg.learning_rate,
                self.train_cfg.min_learning_rate,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            skip_step = False
            skip_reason = ""
            for _ in range(self.train_cfg.grad_accum_steps):
                x, y = self._batch("train", self.train_cfg.micro_batch_size)
                with self._autocast():
                    out = self.model(x, labels=y)
                    loss = out.loss / self.train_cfg.grad_accum_steps
                if not torch.isfinite(loss.detach()):
                    skip_step = True
                    skip_reason = "nonfinite_loss"
                    break
                self.scaler.scale(loss).backward()
                total_loss += float(loss.detach().cpu()) * self.train_cfg.grad_accum_steps
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
                    self.scaler.update(max(float(self.scaler.get_scale()) * 0.5, 1.0))
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.ema and not skip_step:
                self.ema.update(self.model)

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
                    save_checkpoint(
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
                save_checkpoint(
                    self.output_dir / f"step-{step}.pt",
                    self.model,
                    self.optimizer,
                    self.config,
                    step,
                    None,
                    self.ema.state_dict() if self.ema else None,
                )

        save_checkpoint(
            self.output_dir / "last.pt",
            self.model,
            self.optimizer,
            self.config,
            self.train_cfg.max_steps,
            None,
            self.ema.state_dict() if self.ema else None,
        )
        self._log({"train/final_step": self.train_cfg.max_steps, "val/best_loss": best_val}, self.train_cfg.max_steps, event="done")


def train_from_config(path: str | Path) -> None:
    cfg = load_config(path)
    Trainer(cfg).train()
