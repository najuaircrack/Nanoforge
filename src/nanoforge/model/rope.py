from __future__ import annotations

import math
from typing import Any

import torch


def _capturing_graph():
    return (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
            or (hasattr(torch, "compiler") and torch.compiler.is_compiling()))


def validate_rope_scaling(scaling: dict[str, Any] | None) -> None:
    if scaling is None:
        return
    if not isinstance(scaling, dict):
        raise ValueError("RoPE scaling must be a mapping.")
    kind = scaling.get("type", "linear")
    if kind not in {"linear", "dynamic", "yarn"}:
        raise ValueError(f"Unknown rope scaling type: {kind}")
    factor = float(scaling.get("factor", 1.0))
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("RoPE scaling factor must be finite and >= 1.")
    original = float(scaling.get("original_max_position_embeddings", 1))
    if not math.isfinite(original) or original < 1:
        raise ValueError("RoPE original_max_position_embeddings must be >= 1.")


def _scaled_positions(seq_len: int, device: torch.device, scaling: dict[str, Any] | None) -> torch.Tensor:
    validate_rope_scaling(scaling)
    if not _capturing_graph() and (not isinstance(seq_len, int) or seq_len < 0):
        raise ValueError("RoPE sequence length must be a nonnegative integer.")
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    if not scaling:
        return pos
    kind = scaling.get("type", "linear")
    factor = float(scaling.get("factor", 1.0))
    if factor == 1:
        return pos
    if kind == "linear":
        return pos / factor
    if "original_max_position_embeddings" not in scaling:
        return pos
    original = float(scaling["original_max_position_embeddings"])
    if kind == "dynamic":
        length = torch.as_tensor(seq_len, device=device, dtype=torch.float32)
        scale = ((original - 1) / (length - 1).clamp_min(1)).clamp_max(1)
        return pos * scale
    ramp = torch.clamp((pos - original) / max(original * (factor - 1.0), 1.0), 0.0, 1.0)
    return pos / (1.0 + ramp * (factor - 1.0))


class RotaryEmbedding:
    """One growable table per device/dtype; dynamic positions rebuild that table."""

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0,
                 scaling: dict[str, Any] | None = None):
        if dim <= 0 or dim % 2:
            raise ValueError("RoPE head dimension must be positive and even.")
        if max_seq_len <= 0 or not math.isfinite(base) or base <= 0:
            raise ValueError("RoPE maximum length and frequency base must be positive.")
        validate_rope_scaling(scaling)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling = None if scaling is None else dict(scaling)
        self._cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_cos_sin(self, seq_len: int, device: torch.device,
                    dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if not _capturing_graph() and seq_len < 0:
            raise ValueError("RoPE sequence length must be nonnegative.")
        if _capturing_graph():
            pos = _scaled_positions(seq_len, device, self.scaling)
            inv_freq = 1.0 / (self.base ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
            freqs = torch.outer(pos, inv_freq)
            return freqs.cos().to(dtype), freqs.sin().to(dtype)
        key = (device, dtype)
        cached = self._cache.get(key)
        dependent = bool(self.scaling and self.scaling.get("type") == "dynamic")
        if self.scaling and self.scaling.get("type") == "yarn":
            dependent = "original_max_position_embeddings" not in self.scaling
        if cached is not None:
            length = cached[0].shape[0]
            if (dependent and length == seq_len) or (not dependent and length >= seq_len):
                return cached[0][:seq_len], cached[1][:seq_len]
        length = seq_len if dependent else max(seq_len, min(self.max_seq_len,
                    max(16, 2 * (cached[0].shape[0] if cached is not None else 0))))
        pos = _scaled_positions(length, device, self.scaling)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        freqs = torch.outer(pos, inv_freq)
        cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
        self._cache[key] = (cos, sin)
        return cos[:seq_len], sin[:seq_len]

    def get_cos_sin_range(self, start: int, end: int, device: torch.device,
                          dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute only retained absolute positions after cache eviction."""
        if start < 0 or end < start:
            raise ValueError("Invalid rotary position range.")
        pos = torch.arange(start, end, device=device, dtype=torch.float32)
        if self.scaling and float(self.scaling.get("factor", 1.)) != 1:
            factor = float(self.scaling.get("factor", 1.))
            kind = self.scaling.get("type", "linear")
            original = float(self.scaling.get("original_max_position_embeddings", end))
            if kind == "linear":
                pos = pos / factor
            elif kind == "dynamic" and end > original:
                pos = pos * (original - 1) / max(end - 1, 1)
            elif kind == "yarn":
                ramp = ((pos - original) / max(original * (factor - 1), 1)).clamp(0, 1)
                pos = pos / (1 + ramp * (factor - 1))
        inv_freq = 1. / (self.base ** (torch.arange(0, self.dim, 2, device=device,
                                        dtype=torch.float32) / self.dim))
        freqs = torch.outer(pos, inv_freq)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def clear_cache(self) -> None:
        self._cache.clear()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 offset: int = 0) -> torch.Tensor:
    seq_len = x.shape[-2]
    if offset < 0 or offset + seq_len > cos.shape[0]:
        raise ValueError("Rotary table does not cover the requested absolute positions.")
    cos = cos[offset:offset + seq_len][None, None, :, :]
    sin = sin[offset:offset + seq_len][None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def yarn_recommended_theta(base: float, factor: float, head_dim: int) -> float:
    return base * (factor ** (head_dim / max(head_dim - 2, 1)))
