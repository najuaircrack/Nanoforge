from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class HealthEvent:
    kind: str
    severity: str
    message: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class HealthSnapshot:
    metrics: dict[str, float]
    events: list[HealthEvent]


class TrainingHealthMonitor:
    """Low-overhead diagnostics for long local training runs."""

    def __init__(self, *, grad_explosion_factor: float = 10.0, history: int = 128):
        self.grad_explosion_factor = grad_explosion_factor
        self.history = history
        self.losses: list[float] = []
        self.grad_norms: list[float] = []

    def observe(
        self,
        *,
        loss: float,
        grad_norm: float,
        logits: torch.Tensor | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | None = None,
    ) -> HealthSnapshot:
        events: list[HealthEvent] = []
        metrics: dict[str, float] = {}
        if math.isfinite(loss):
            self.losses.append(loss)
            self.losses = self.losses[-self.history :]
        if math.isfinite(grad_norm):
            baseline = _median(self.grad_norms[-32:])
            if baseline > 0 and grad_norm > baseline * self.grad_explosion_factor:
                events.append(
                    HealthEvent(
                        "exploding_gradient",
                        "warning",
                        "Gradient norm spiked far above recent history; consider lowering LR or grad_clip.",
                        {"grad_norm": grad_norm, "recent_median_grad_norm": baseline},
                    )
                )
            self.grad_norms.append(grad_norm)
            self.grad_norms = self.grad_norms[-self.history :]
        if self.losses:
            metrics["health/loss_recent_mean"] = sum(self.losses[-32:]) / min(len(self.losses), 32)
        if self.grad_norms:
            metrics["health/grad_norm_recent_median"] = _median(self.grad_norms[-32:])
        if logits is not None:
            entropy = token_entropy(logits)
            metrics["health/logit_entropy"] = entropy
            if entropy < 0.25:
                events.append(
                    HealthEvent(
                        "entropy_collapse",
                        "warning",
                        "Logit entropy is very low; generation may collapse or repeat.",
                        {"logit_entropy": entropy},
                    )
                )
        if optimizer is not None:
            metrics.update(optimizer_state_stats(optimizer))
        if device is not None:
            metrics.update(memory_stats(device))
        return HealthSnapshot(metrics=metrics, events=events)


@torch.no_grad()
def token_entropy(logits: torch.Tensor) -> float:
    last = logits[:, -1, :].detach().float()
    probs = torch.softmax(last, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return float(entropy.cpu())


def optimizer_state_stats(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    tensors = 0
    elements = 0
    nonfinite = 0
    for state in optimizer.state.values():
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            tensors += 1
            elements += value.numel()
            if value.dtype.is_floating_point and not torch.isfinite(value).all():
                nonfinite += 1
    return {
        "health/optimizer_state_tensors": float(tensors),
        "health/optimizer_state_elements": float(elements),
        "health/optimizer_nonfinite_tensors": float(nonfinite),
    }


def memory_stats(device: torch.device) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if device.type == "cuda" and torch.cuda.is_available():
        metrics["memory/cuda_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
        metrics["memory/cuda_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)
        metrics["memory/cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return metrics


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])
