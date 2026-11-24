from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import partial

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
        activation_metrics: dict[str, float] | None = None,
    ) -> HealthSnapshot:
        events: list[HealthEvent] = []
        metrics: dict[str, float] = {}
        metrics.update(activation_metrics or {})
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

    def state_dict(self):
        return {"losses": list(self.losses), "grad_norms": list(self.grad_norms)}

    def load_state_dict(self, state):
        self.losses = [float(value) for value in state.get("losses", [])][-self.history :]
        self.grad_norms = [float(value) for value in state.get("grad_norms", [])][-self.history :]


@torch.no_grad()
def token_entropy(logits: torch.Tensor) -> float:
    last = (logits[:, -1, :] if logits.ndim == 3 else logits).detach().float()
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
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess
            process.restype = wintypes.HANDLE
            memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            # The buffer has the documented PROCESS_MEMORY_COUNTERS layout. A
            # void pointer avoids cached ctypes function signatures referencing a
            # newly defined local Structure type on another diagnostic thread.
            memory_info.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
            memory_info.restype = wintypes.BOOL
            if memory_info(process(), ctypes.byref(counters), counters.cb):
                metrics["memory/process_rss_mb"] = counters.WorkingSetSize / (1024 * 1024)
                metrics["memory/process_peak_rss_mb"] = counters.PeakWorkingSetSize / (1024 * 1024)
        else:
            import resource
            import sys

            metrics["memory/process_peak_rss_mb"] = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except (ImportError, OSError, AttributeError):
        pass
    if device.type == "cuda" and torch.cuda.is_available():
        metrics["memory/cuda_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)
        metrics["memory/cuda_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 * 1024)
        metrics["memory/cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (
            1024 * 1024
        )
        statistics = torch.cuda.memory_stats(device)
        metrics["memory/cuda_inactive_split_mb"] = statistics.get(
            "inactive_split_bytes.all.current", 0
        ) / (1024 * 1024)
    return metrics


class ActivationDiagnostics:
    """Aggregate scalar diagnostics at configured intervals; never retain activations or input text."""

    def __init__(self, model: torch.nn.Module):
        self.enabled = False
        self.values = []
        self.nonfinite_modules = []
        self.handles = [
            module.register_forward_hook(partial(self._observe, name))
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]

    @torch.no_grad()
    def _observe(self, name, module, inputs, output):
        if not self.enabled or not torch.is_tensor(output) or output.numel() == 0:
            return
        values = output.detach().float().reshape(-1, output.shape[-1])
        finite = torch.isfinite(values)
        if not bool(finite.all()) and name not in self.nonfinite_modules and len(self.nonfinite_modules) < 8:
            self.nonfinite_modules.append(name)
        safe = torch.where(finite, values, torch.zeros_like(values))
        self.values.append(
            {
                "health/activation_mean": float(safe.mean()),
                "health/activation_std": float(safe.std(unbiased=False)),
                "health/activation_nonfinite_fraction": float((~finite).float().mean()),
                "health/activation_saturation_fraction": float((safe.abs() > 8).float().mean()),
                "health/dead_channel_fraction": float(
                    (safe.square().mean(0) < 1e-12).float().mean()
                ),
            }
        )

    def begin(self, enabled: bool):
        self.enabled = enabled
        self.values.clear()
        self.nonfinite_modules.clear()

    def summary(self):
        return (
            {
                key: sum(value[key] for value in self.values) / len(self.values)
                for key in self.values[0]
            }
            if self.values
            else {}
        )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])
