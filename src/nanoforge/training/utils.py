from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if precision == "fp32" or device.type == "cpu":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def cosine_lr(step: int, max_steps: int, warmup_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + coeff * (lr - min_lr)


def configure_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float = 1e-8,
):
    return configure_named_optimizer(model, "adamw", lr, weight_decay, betas, eps)


def configure_named_optimizer(
    model: torch.nn.Module,
    name: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float = 1e-8,
):
    from nanoforge.training.optimizers import create_optimizer

    optimizer_name = name
    decay, no_decay = [], []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "embed" not in param_name:
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return create_optimizer(optimizer_name, groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


@torch.no_grad()
def grad_global_norm(parameters) -> float:
    total = torch.zeros((), dtype=torch.float32)
    found = False
    for param in parameters:
        if param.grad is None:
            continue
        found = True
        grad = param.grad.detach().float()
        total += grad.pow(2).sum()
    if not found:
        return 0.0
    return float(total.sqrt().cpu())


def grads_are_finite(parameters) -> bool:
    for param in parameters:
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad and p.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_low_memory_env() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
