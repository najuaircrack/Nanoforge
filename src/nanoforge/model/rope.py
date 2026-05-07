from __future__ import annotations

import math
from typing import Any

import torch


def _scaled_positions(seq_len: int, device: torch.device, scaling: dict[str, Any] | None) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    if not scaling:
        return pos
    kind = scaling.get("type", "linear")
    factor = float(scaling.get("factor", 1.0))
    if factor <= 1.0:
        return pos
    if kind == "linear":
        return pos / factor
    if kind == "dynamic":
        original = float(scaling.get("original_max_position_embeddings", seq_len))
        if seq_len <= original:
            return pos
        return pos * (original - 1) / max(seq_len - 1, 1)
    if kind == "yarn":
        original = float(scaling.get("original_max_position_embeddings", seq_len))
        ramp = torch.clamp((pos - original) / max(original * (factor - 1.0), 1.0), 0.0, 1.0)
        return pos / (1.0 + ramp * (factor - 1.0))
    raise ValueError(f"Unknown rope scaling type: {kind}")


class RotaryEmbedding:
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        base: float = 10000.0,
        scaling: dict[str, Any] | None = None,
    ):
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling = scaling
        self._cache: dict[tuple[torch.device, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype, seq_len)
        if key in self._cache:
            return self._cache[key]
        pos = _scaled_positions(seq_len, device, self.scaling)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        freqs = torch.outer(pos, inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        self._cache[key] = (cos, sin)
        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    # x: [batch, heads, seq, head_dim]
    seq_len = x.shape[-2]
    cos = cos[offset : offset + seq_len][None, None, :, :]
    sin = sin[offset : offset + seq_len][None, None, :, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def yarn_recommended_theta(base: float, factor: float, head_dim: int) -> float:
    return base * (factor ** (head_dim / max(head_dim - 2, 1)))

