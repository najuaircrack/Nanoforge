from __future__ import annotations

import os
import random
from contextlib import contextmanager
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
    return create_optimizer(
        optimizer_name, groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    )


@torch.no_grad()
def grad_global_norm(parameters) -> float:
    squared_norms = []
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        squared_norms.append(grad.pow(2).sum())
    if not squared_norms:
        return 0.0
    device = squared_norms[0].device
    return float(torch.stack([value.to(device) for value in squared_norms]).sum().sqrt().cpu())


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

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if set(state) != set(self.shadow):
            raise ValueError("Averaged-weight state does not match model parameters.")
        for name, value in state.items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"Averaged weight shape mismatch: {name}")
            self.shadow[name].copy_(value.to(self.shadow[name].device))

    @contextmanager
    def applied(self, model: torch.nn.Module):
        parameters = dict(model.named_parameters())
        original = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in original.items():
                    parameters[name].copy_(value)


class SWA(EMA):
    """Uniform average of explicit post-update snapshots; never replaces live optimizer weights."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model, decay=0.0)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 / self.updates)


def move_optimizer_state(optimizer: torch.optim.Optimizer, *, to_cpu: bool) -> None:
    """Offload persistent state between updates; transfer it back for the optimizer step."""

    def move(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item, device) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item, device) for item in value)
        return value

    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = move(
            state, torch.device("cpu") if to_cpu else parameter.device
        )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_low_memory_env() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
