from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nanoforge.config import ModelConfig
from nanoforge.model.lora import LoRALinear
from nanoforge.model.rope import RotaryEmbedding, apply_rotary


@dataclass
class KVCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    max_length: int | None = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key is None:
            self.key = key
            self.value = value
        else:
            self.key = torch.cat([self.key, key], dim=2)
            self.value = torch.cat([self.value, value], dim=2)
        if self.max_length is not None and self.key.shape[2] > self.max_length:
            self.key = self.key[:, :, -self.max_length :, :].contiguous()
            self.value = self.value[:, :, -self.max_length :, :].contiguous()
        return self.key, self.value

    @property
    def length(self) -> int:
        return 0 if self.key is None else int(self.key.shape[2])

    def clone_detached(self) -> "KVCache":
        return KVCache(
            None if self.key is None else self.key.detach().clone(),
            None if self.value is None else self.value.detach().clone(),
            self.max_length,
        )


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, heads, seq, dim = x.shape
    x = x[:, :, None, :, :].expand(batch, heads, n_rep, seq, dim)
    return x.reshape(batch, heads * n_rep, seq, dim)


def _attention_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
    sliding_window: int | None,
) -> torch.Tensor:
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
    k_pos = torch.arange(kv_len, device=device)[None, :]
    allowed = k_pos <= q_pos
    if sliding_window is not None:
        allowed = allowed & (k_pos >= q_pos - sliding_window + 1)
    mask = torch.full((q_len, kv_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    return mask.masked_fill(allowed, 0.0)


def _alibi_slopes(n_heads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    def power_of_two_slopes(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = power_of_two_slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = power_of_two_slopes(closest)
        slopes += power_of_two_slopes(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(slopes, device=device, dtype=dtype).view(1, n_heads, 1, 1)


def _alibi_bias(q_len: int, kv_len: int, n_heads: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device, dtype=dtype)[:, None]
    k_pos = torch.arange(kv_len, device=device, dtype=dtype)[None, :]
    distance = (q_pos - k_pos).clamp_min(0)
    return -_alibi_slopes(n_heads, device, dtype) * distance.view(1, 1, q_len, kv_len)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.kv_repeat = self.n_heads // self.n_kv_heads
        self.use_flash = config.use_flash
        rank = config.lora_rank
        self.q_proj = LoRALinear(
            config.d_model,
            self.n_heads * self.head_dim,
            rank=rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        self.k_proj = LoRALinear(
            config.d_model,
            self.n_kv_heads * self.head_dim,
            rank=rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        self.v_proj = LoRALinear(
            config.d_model,
            self.n_kv_heads * self.head_dim,
            rank=rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        self.o_proj = LoRALinear(
            self.n_heads * self.head_dim,
            config.d_model,
            rank=rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(
            self.head_dim, config.max_seq_len, config.rope_theta, config.rope_scaling
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0 if cache is None or cache.key is None else cache.key.shape[2]
        if self.config.position_embedding == "rope":
            cos, sin = self.rope.get_cos_sin(past_len + seq_len, x.device, q.dtype)
            q = apply_rotary(q, cos, sin, past_len)
            k = apply_rotary(k, cos, sin, past_len)

        if use_cache:
            cache = cache or KVCache()
            cache.max_length = self.config.sliding_window or self.config.max_seq_len
            k, v = cache.append(k, v)
        kv_len = k.shape[2]
        k_full = repeat_kv(k, self.kv_repeat)
        v_full = repeat_kv(v, self.kv_repeat)

        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            is_simple_causal = (
                kv_len == seq_len
                and self.config.sliding_window is None
                and self.config.position_embedding != "alibi"
            )
            attn_mask = None
            if not is_simple_causal:
                attn_mask = _attention_mask(
                    seq_len, kv_len, x.device, q.dtype, self.config.sliding_window
                )
            if self.config.position_embedding == "alibi":
                bias = _alibi_bias(seq_len, kv_len, self.n_heads, x.device, q.dtype)
                attn_mask = bias if attn_mask is None else attn_mask + bias
            y = F.scaled_dot_product_attention(
                q,
                k_full,
                v_full,
                attn_mask=attn_mask,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=is_simple_causal,
            )
        else:
            scores = (q @ k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + _attention_mask(
                seq_len, kv_len, x.device, scores.dtype, self.config.sliding_window
            )
            if self.config.position_embedding == "alibi":
                scores = scores + _alibi_bias(seq_len, kv_len, self.n_heads, x.device, scores.dtype)
            probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
            probs = self.dropout(probs)
            y = probs @ v_full
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(y), cache


class ChunkedCausalSelfAttention(CausalSelfAttention):
    """Memory-bounded attention variant using a local chunk-sized cache window."""

    def __init__(self, config: ModelConfig):
        cfg = copy(config)
        cfg.sliding_window = cfg.sliding_window or min(cfg.max_seq_len, 1024)
        super().__init__(cfg)


class SparseCausalSelfAttention(CausalSelfAttention):
    """Experimental sparse-style local attention approximation."""

    def __init__(self, config: ModelConfig):
        cfg = copy(config)
        cfg.sliding_window = cfg.sliding_window or min(cfg.max_seq_len, 512)
        super().__init__(cfg)


class HybridLocalGlobalCausalAttention(CausalSelfAttention):
    """Hybrid backend hook with local-window behavior and full-prefill compatibility."""

    def __init__(self, config: ModelConfig):
        cfg = copy(config)
        cfg.sliding_window = cfg.sliding_window or min(cfg.max_seq_len, 2048)
        super().__init__(cfg)


class PagedCausalSelfAttention(CausalSelfAttention):
    """Paged-cache compatible attention surface.

    The current implementation uses bounded contiguous pages internally; the class gives runtime
    code a distinct backend to target without pretending it is the base SDPA backend.
    """

    def __init__(self, config: ModelConfig):
        cfg = copy(config)
        cfg.sliding_window = cfg.sliding_window or cfg.max_seq_len
        super().__init__(cfg)
