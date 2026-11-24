from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nanoforge.config import ModelConfig
from nanoforge.model.lora import LoRALinear
from nanoforge.model.rope import RotaryEmbedding, _capturing_graph, apply_rotary

_log = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_func as _flash_attn_func
except (ImportError, OSError):
    _flash_attn_func = None
_FLASH_ATTN_AVAILABLE = _flash_attn_func is not None


@dataclass
class KVCache:
    """Bounded unrotated K/V plus logical position and optional sequence-mixer state."""

    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    max_length: int | None = None
    tokens_seen: int = 0
    ffn_state: torch.Tensor | None = None

    def __post_init__(self):
        if self.tokens_seen == 0 and self.key is not None:
            self.tokens_seen = self.key.shape[2]

    def append(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_pair(key, value)
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("KV cache maximum length must be positive.")
        if self.key is not None:
            _validate_compatible(self.key, key)
            key, value = torch.cat((self.key, key), 2), torch.cat((self.value, value), 2)
        added = key.shape[2] - self.length
        self.tokens_seen += added
        self.key, self.value = key, value
        if self.max_length is not None and self.length > self.max_length:
            self.key = key[:, :, -self.max_length:].contiguous()
            self.value = value[:, :, -self.max_length:].contiguous()
        return self.key, self.value

    @property
    def length(self) -> int:
        return 0 if self.key is None else self.key.shape[2]

    @property
    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.key, self.value, self.ffn_state)
                   if t is not None)

    def iter_blocks(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self.key is not None:
            yield self.key, self.value

    def clone_detached(self) -> KVCache:
        return KVCache(None if self.key is None else self.key.detach().clone(),
                       None if self.value is None else self.value.detach().clone(),
                       self.max_length, self.tokens_seen,
                       None if self.ffn_state is None else self.ffn_state.detach().clone())


class PagedKVCache(KVCache):
    """CPU/PyTorch reference cache with independent bounded-size tensor pages."""

    def __init__(self, max_length: int | None = None, page_size: int = 64):
        if page_size <= 0 or (max_length is not None and max_length <= 0):
            raise ValueError("Page size and cache maximum length must be positive.")
        self.pages: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.page_size, self.max_length = page_size, max_length
        self.tokens_seen = 0
        self.ffn_state = None

    @property
    def key(self):
        return None if not self.pages else torch.cat([k for k, _ in self.pages], 2)

    @property
    def value(self):
        return None if not self.pages else torch.cat([v for _, v in self.pages], 2)

    @property
    def length(self):
        return sum(k.shape[2] for k, _ in self.pages)

    @property
    def storage_bytes(self):
        result = sum(t.numel() * t.element_size() for pair in self.pages for t in pair)
        if self.ffn_state is not None:
            result += self.ffn_state.numel() * self.ffn_state.element_size()
        return result

    def iter_blocks(self):
        yield from self.pages

    def append(self, key, value):
        _validate_pair(key, value)
        if self.pages:
            _validate_compatible(self.pages[0][0], key)
        self.tokens_seen += key.shape[2]
        position = 0
        if self.pages and self.pages[-1][0].shape[2] < self.page_size:
            old_k, old_v = self.pages[-1]
            width = min(self.page_size - old_k.shape[2], key.shape[2])
            self.pages[-1] = (torch.cat((old_k, key[:, :, :width]), 2),
                              torch.cat((old_v, value[:, :, :width]), 2))
            position = width
        for start in range(position, key.shape[2], self.page_size):
            self.pages.append((key[:, :, start:start + self.page_size].contiguous().clone(),
                               value[:, :, start:start + self.page_size].contiguous().clone()))
        if self.max_length is not None:
            excess = self.length - self.max_length
            while excess > 0:
                old_k, old_v = self.pages[0]
                if excess >= old_k.shape[2]:
                    excess -= old_k.shape[2]
                    self.pages.pop(0)
                else:
                    self.pages[0] = (old_k[:, :, excess:].contiguous().clone(),
                                     old_v[:, :, excess:].contiguous().clone())
                    excess = 0
        # No whole-cache materialization on the paged execution path.
        return None, None

    def clone_detached(self):
        result = PagedKVCache(self.max_length, self.page_size)
        result.pages = [(k.detach().clone(), v.detach().clone()) for k, v in self.pages]
        result.tokens_seen = self.tokens_seen
        result.ffn_state = None if self.ffn_state is None else self.ffn_state.detach().clone()
        return result


def _validate_pair(key, value):
    if key.ndim != 4 or value.shape != key.shape or not key.is_floating_point():
        raise ValueError("KV cache tensors must be matching floating [batch, heads, tokens, dim].")
    if key.device != value.device or key.dtype != value.dtype:
        raise ValueError("KV cache key/value device and dtype must match.")


def _validate_compatible(old, new):
    if (old.shape[:2] != new.shape[:2] or old.shape[3] != new.shape[3]
            or old.dtype != new.dtype or old.device != new.device):
        raise ValueError("KV cache batch, heads, dimension, dtype or device mismatch.")


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, heads, seq, dim = x.shape
    return x[:, :, None].expand(batch, heads, n_rep, seq, dim).reshape(batch, heads * n_rep, seq, dim)


def _attention_mask(q_len, kv_len, device, dtype, sliding_window):
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
    k_pos = torch.arange(kv_len, device=device)[None]
    allowed = k_pos <= q_pos
    if sliding_window is not None:
        allowed &= k_pos >= q_pos - sliding_window + 1
    return torch.zeros((q_len, kv_len), device=device, dtype=dtype).masked_fill(~allowed, -torch.inf)


def _alibi_slopes(n_heads, device, dtype):
    def powers(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]
    if math.log2(n_heads).is_integer():
        slopes = powers(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = powers(closest) + powers(2 * closest)[0::2][:n_heads - closest]
    return torch.tensor(slopes, device=device, dtype=dtype).view(1, n_heads, 1, 1)


def _alibi_bias(q_len, kv_len, n_heads, device, dtype):
    q = torch.arange(kv_len - q_len, kv_len, device=device, dtype=dtype)[:, None]
    k = torch.arange(kv_len, device=device, dtype=dtype)[None]
    return -_alibi_slopes(n_heads, device, dtype) * (q - k).clamp_min(0)[None, None]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.kv_repeat = self.n_heads // self.n_kv_heads
        self.use_flash = config.use_flash
        self.actual_backend = "unselected"
        self.fallback_reason: str | None = None
        self.collect_diagnostics = False
        self.last_attention_diagnostics = None
        self.chunk_size = getattr(config, "attention_chunk_size", 128)
        if self.chunk_size <= 0 or (config.sliding_window is not None and config.sliding_window <= 0):
            raise ValueError("Attention chunk size and window must be positive.")
        for name, out_features in (("q_proj", self.n_heads * self.head_dim),
                                   ("k_proj", self.n_kv_heads * self.head_dim),
                                   ("v_proj", self.n_kv_heads * self.head_dim),
                                   ("o_proj", config.d_model)):
            setattr(self, name, LoRALinear(config.d_model, out_features, rank=config.lora_rank,
                    alpha=config.lora_alpha, dropout=config.lora_dropout))
        self.dropout = nn.Dropout(config.dropout)
        self.rope = (RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta,
                                   config.rope_scaling) if config.position_embedding == "rope" else None)

    def make_cache(self):
        return KVCache(max_length=self.config.sliding_window or self.config.max_seq_len)

    def allowed(self, q_pos, k_pos, window):
        allowed = k_pos <= q_pos if self.config.is_causal else torch.ones_like(q_pos + k_pos, dtype=torch.bool)
        if window is not None:
            if self.config.is_causal:
                allowed &= k_pos >= q_pos - window + 1
            else:
                allowed &= (q_pos - k_pos).abs() < window
        return allowed

    def mask(self, q_start, q_length, k_start, k_length, device, dtype, window):
        q = torch.arange(q_start, q_start + q_length, device=device)[:, None]
        k = torch.arange(k_start, k_start + k_length, device=device)[None]
        allowed = self.allowed(q, k, window)
        mask = torch.zeros(allowed.shape, device=device, dtype=dtype).masked_fill(~allowed, -torch.inf)
        if self.config.position_embedding == "alibi":
            mask = mask[None, None] - _alibi_slopes(self.n_heads, device, dtype) * (q - k).abs()[None, None]
        return mask

    def forward(self, x, cache=None, use_cache=False):
        if x.ndim != 3 or x.shape[1] == 0 or x.shape[2] != self.config.d_model:
            raise ValueError("Attention input must be nonempty [batch, tokens, d_model].")
        if cache is not None and not isinstance(cache, KVCache):
            raise ValueError("Invalid KV cache type.")
        if cache is not None and not use_cache:
            raise ValueError("A cache requires use_cache=True.")
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        raw_k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cache = (cache or self.make_cache()) if use_cache else None
        past = 0 if cache is None else cache.tokens_seen
        old_length = 0 if cache is None else cache.length
        blocks = list(cache.iter_blocks()) if cache is not None else []
        if blocks:
            _validate_compatible(blocks[0][0], raw_k)
        blocks.append((raw_k, v))
        start = past - old_length
        if start < 0:
            raise ValueError("Cache logical position precedes its retained storage.")
        if self.rope is not None:
            if start > 0:
                cos, sin = self.rope.get_cos_sin_range(start, past + seq_len, x.device, q.dtype)
                origin = start
            else:
                cos, sin = self.rope.get_cos_sin(past + seq_len, x.device, q.dtype)
                origin = 0
            q = apply_rotary(q, cos, sin, past - origin)
            rotated = []
            position = start - origin
            for key, value in blocks:
                rotated.append((apply_rotary(key, cos, sin, position), value))
                position += key.shape[2]
            blocks = rotated
        window = self.config.sliding_window
        if use_cache and window is None:
            window = self.config.max_seq_len
        y = self.attend(q, blocks, past, start, window)
        self.last_attention_diagnostics = (self.diagnose(q, blocks, past, start, window)
                                           if self.collect_diagnostics else None)
        if cache is not None:
            cache.append(raw_k, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(y), cache

    @torch.no_grad()
    def diagnose(self, q, blocks, q_start, k_start, window):
        query = q[:, :, -1:].float()
        scores, position = [], k_start
        for key, _ in blocks:
            repeated = repeat_kv(key, self.kv_repeat).float()
            mask = self.mask(q_start + q.shape[2] - 1, 1, position, key.shape[2],
                             q.device, torch.float32, window)
            scores.append(query @ repeated.transpose(-2, -1) / math.sqrt(self.head_dim) + mask)
            position += key.shape[2]
        probabilities = torch.softmax(torch.cat(scores, -1), -1)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
        return {"entropy_nats_mean": float(entropy.mean()),
                "entropy_nats_min": float(entropy.min()),
                "entropy_nats_max": float(entropy.max()),
                "scope": "reference_last_query_pre_dropout",
                "query_count": 1, "key_count": position - k_start,
                "nonfinite_fraction": float((~torch.isfinite(probabilities)).float().mean())}

    def attend(self, q, blocks, q_start, k_start, window):
        k = repeat_kv(torch.cat([key for key, _ in blocks], 2), self.kv_repeat)
        v = repeat_kv(torch.cat([value for _, value in blocks], 2), self.kv_repeat)
        requested = self.config.attention_backend
        external = requested in {"flash", "flash_v2"}
        simple = q.shape[2] == k.shape[2] and (window is None or window >= k.shape[2])
        eligible = (external and not _capturing_graph() and self.use_flash and _FLASH_ATTN_AVAILABLE and q.is_cuda
                    and q.dtype in (torch.float16, torch.bfloat16)
                    and self.config.position_embedding != "alibi" and simple)
        if eligible:
            self.actual_backend, self.fallback_reason = "flash_attn", None
            return _flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    dropout_p=self.config.dropout if self.training else 0.,
                    causal=bool(self.config.is_causal)).transpose(1, 2)
        mask = self.mask(q_start, q.shape[2], k_start, k.shape[2], q.device, q.dtype, window)
        manual = requested == "manual" or not self.use_flash or not hasattr(F, "scaled_dot_product_attention")
        self.actual_backend = "manual" if manual else "sdpa"
        self.fallback_reason = (f"{requested} unavailable for this device, dtype, window or cached shape"
                                if requested.startswith("flash") else None)
        if manual:
            scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(self.head_dim)
            weights = torch.softmax(scores + mask.float(), dim=-1).nan_to_num(0).to(q.dtype)
            return self.dropout(weights) @ v
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                    dropout_p=self.config.dropout if self.training else 0., is_causal=False)

    def reference_attention(self, q, blocks, q_start, k_start, window):
        """Differentiable online softmax over query chunks and key/value pages."""
        outputs = []
        for begin in range(0, q.shape[2], self.chunk_size):
            query = q[:, :, begin:begin + self.chunk_size].float()
            maximum = torch.full((*query.shape[:-1], 1), -torch.inf, device=q.device)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros_like(query)
            position = k_start
            for key, value in blocks:
                for offset in range(0, key.shape[2], self.chunk_size):
                    key_part = repeat_kv(key[:, :, offset:offset + self.chunk_size], self.kv_repeat).float()
                    value_part = repeat_kv(value[:, :, offset:offset + self.chunk_size], self.kv_repeat).float()
                    mask = self.mask(q_start + begin, query.shape[2], position + offset,
                                    key_part.shape[2], q.device, torch.float32, window)
                    if not torch.isfinite(mask).any():
                        continue
                    scores = query @ key_part.transpose(-2, -1) / math.sqrt(self.head_dim) + mask
                    next_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
                    safe_max = next_max.masked_fill(~torch.isfinite(next_max), 0.)
                    alpha = torch.exp(maximum - safe_max).masked_fill(~torch.isfinite(maximum), 0.)
                    weight = torch.exp(scores - safe_max).masked_fill(~torch.isfinite(scores), 0.)
                    denominator = alpha * denominator + weight.sum(-1, keepdim=True)
                    numerator = alpha * numerator + F.dropout(weight,
                        p=self.config.dropout if self.training else 0., training=self.training) @ value_part
                    maximum = next_max
                position += key.shape[2]
            outputs.append((numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)).to(q.dtype))
        return torch.cat(outputs, 2)


class ChunkedCausalSelfAttention(CausalSelfAttention):
    """Exact dense attention using bounded query/key tiles and online softmax."""

    def attend(self, q, blocks, q_start, k_start, window):
        if _capturing_graph():
            output = super().attend(q, blocks, q_start, k_start, window)
            self.actual_backend = "dense_export_equivalent"
            return output
        self.actual_backend, self.fallback_reason = "chunked_reference", None
        return self.reference_attention(q, blocks, q_start, k_start, window)


class SparseCausalSelfAttention(ChunkedCausalSelfAttention):
    """Reference local sparse attention; fully excluded key blocks are skipped."""

    def __init__(self, config):
        cfg = copy(config)
        cfg.sliding_window = cfg.sliding_window or min(cfg.max_seq_len, 512)
        super().__init__(cfg)

    def attend(self, *args):
        result = super().attend(*args)
        self.actual_backend = "dense_export_equivalent" if _capturing_graph() else "sparse_local_reference"
        return result


class HybridLocalGlobalCausalAttention(ChunkedCausalSelfAttention):
    """Periodic global queries interspersed with local queries."""

    def __init__(self, config):
        cfg = copy(config)
        self.local_window = cfg.sliding_window or min(cfg.max_seq_len, 512)
        self.global_stride = getattr(cfg, "hybrid_global_stride", 64)
        if self.global_stride <= 0:
            raise ValueError("Hybrid global stride must be positive.")
        cfg.sliding_window = None
        super().__init__(cfg)

    def allowed(self, q_pos, k_pos, window):
        base = super().allowed(q_pos, k_pos, window)
        local = k_pos >= q_pos - self.local_window + 1
        return base & (local | (q_pos.remainder(self.global_stride) == 0))

    def attend(self, *args):
        result = super().attend(*args)
        self.actual_backend = "dense_export_equivalent" if _capturing_graph() else "hybrid_local_global_reference"
        return result


class PagedCausalSelfAttention(ChunkedCausalSelfAttention):
    """Paged CPU/PyTorch reference; no contiguous whole-cache attention materialization."""

    def make_cache(self):
        return PagedKVCache(self.config.sliding_window or self.config.max_seq_len,
                            getattr(self.config, "kv_page_size", 64))

    def attend(self, *args):
        result = super().attend(*args)
        self.actual_backend = "dense_export_equivalent" if _capturing_graph() else "paged_reference"
        return result
