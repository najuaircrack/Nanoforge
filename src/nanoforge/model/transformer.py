from __future__ import annotations

import math

import torch
import torch.utils.checkpoint
from torch import nn

from nanoforge.config import ModelConfig, NanoforgeConfig
from nanoforge.model.attention import KVCache
from nanoforge.model.heads import (  # noqa: F401 (re-export)
    CausalLMHead,
    CausalLMOutput,
    ModelOutput,
)
from nanoforge.registry import (
    ATTENTION_BACKENDS,
    FFN_BACKENDS,
    NORMALIZATIONS,
    TASK_HEADS,
    TRANSFORMER_BLOCKS,
)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.drop(self.embed(input_ids))


def _build_ffn(config: ModelConfig) -> tuple[nn.Module, bool]:
    """Construct the FFN (or MoE) sublayer for a block. Returns (module, is_moe)."""
    hidden = int(config.d_model * config.ffn_hidden_mult)
    hidden = int(256 * math.ceil(hidden / 256)) if hidden >= 256 else hidden
    if config.moe is not None:
        ffn = FFN_BACKENDS.create("moe", config.d_model, hidden, config.moe, config.activation, config.dropout)
        return ffn, True
    ffn_type = config.ffn_type or config.activation
    activation = ffn_type if ffn_type in {"swiglu", "geglu"} else config.activation
    ffn = FFN_BACKENDS.create(ffn_type, config.d_model, hidden, activation, config.dropout)
    return ffn, False


def _run_ffn(ffn, hidden, cache, use_cache):
    if use_cache and hasattr(ffn, "forward_cached"):
        output, state = ffn.forward_cached(hidden, cache.ffn_state)
        cache.ffn_state = state
        return output
    return ffn(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        norm_factory = NORMALIZATIONS.get(config.normalization)
        self.attn_norm = norm_factory(config.d_model, config.norm_eps)
        self.ffn_norm = norm_factory(config.d_model, config.norm_eps)
        self.attn = ATTENTION_BACKENDS.create(config.attention_backend, config)
        self.ffn, self.moe = _build_ffn(config)
        self.residual_scale = config.residual_scale or (1.0 / math.sqrt(2 * config.n_layers))

    def forward(self, x, cache=None, use_cache=False):
        attn_out, cache = self.attn(self.attn_norm(x), cache=cache, use_cache=use_cache)
        x = x + attn_out * self.residual_scale
        ffn_in = self.ffn_norm(x)
        aux_loss = None
        if self.moe:
            ffn_out, aux_loss = self.ffn(ffn_in)
        else:
            ffn_out = _run_ffn(self.ffn, ffn_in, cache, use_cache)
        x = x + ffn_out * self.residual_scale
        return x, cache, aux_loss


class ParallelResidualTransformerBlock(nn.Module):
    """GPT-NeoX style parallel attention/FFN residual block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        norm_factory = NORMALIZATIONS.get(config.normalization)
        self.norm = norm_factory(config.d_model, config.norm_eps)
        self.attn = ATTENTION_BACKENDS.create(config.attention_backend, config)
        self.ffn, self.moe = _build_ffn(config)
        self.residual_scale = config.residual_scale or (1.0 / math.sqrt(2 * config.n_layers))

    def forward(self, x, cache=None, use_cache=False):
        hidden = self.norm(x)
        attn_out, cache = self.attn(hidden, cache=cache, use_cache=use_cache)
        aux_loss = None
        if self.moe:
            ffn_out, aux_loss = self.ffn(hidden)
        else:
            ffn_out = _run_ffn(self.ffn, hidden, cache, use_cache)
        x = x + (attn_out + ffn_out) * self.residual_scale
        return x, cache, aux_loss


class NanoforgeBackbone(nn.Module):
    """Shared transformer backbone: embed → blocks → norm → hidden states."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model, config.dropout)
        self.blocks = nn.ModuleList(
            [TRANSFORMER_BLOCKS.create(config.block_type, config) for _ in range(config.n_layers)]
        )
        self.norm = NORMALIZATIONS.create(config.normalization, config.d_model, config.norm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor, caches=None, use_cache: bool = False):
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("Model input must be nonempty [batch, tokens].")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Model input IDs must be integer tensors.")
        if caches is not None:
            if not isinstance(caches, (list, tuple)) or len(caches) != len(self.blocks):
                raise ValueError("The cache list must have one entry per transformer block.")
            if any(cache is not None and not isinstance(cache, KVCache) for cache in caches):
                raise ValueError("Invalid transformer cache entry.")
            positions = {cache.tokens_seen if cache is not None else 0 for cache in caches}
            if len(positions) > 1:
                raise ValueError("Transformer cache positions must agree across layers.")
        x = self.embedding(input_ids)
        new_caches: list[KVCache] = []
        aux_losses: list[torch.Tensor] = []
        if caches is None:
            caches = [None] * len(self.blocks)

        for block, cache in zip(self.blocks, caches):
            if self.config.gradient_checkpointing and self.training and not use_cache:
                x, new_cache, aux = torch.utils.checkpoint.checkpoint(
                    block, x, None, False, use_reentrant=False
                )
            else:
                x, new_cache, aux = block(x, cache=cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(new_cache or KVCache())
            if aux is not None:
                aux_losses.append(aux)

        x = self.norm(x)
        return x, new_caches or None, aux_losses

    def estimate_num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embedding.embed.weight.numel()
        return n


class NanoforgeModel(nn.Module):
    """Backbone + pluggable task head. Supports any task type registered in TASK_HEADS."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = NanoforgeBackbone(config)

        head_factory = TASK_HEADS.get(config.task_type)
        # Generative heads get the embedding weight for optional tying
        tie_weight = self.backbone.embedding.embed.weight if config.tie_embeddings else None
        try:
            self.head = head_factory(config, tie_weight)
        except TypeError:
            self.head = head_factory(config)

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    # Keep .embed accessible for compat (e.g. engine weight tying checks)
    @property
    def embed(self):
        return self.backbone.embedding.embed

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        caches=None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(f"Sequence length exceeds max_seq_len={self.config.max_seq_len}.")
        hidden, new_caches, aux_losses = self.backbone(input_ids, caches, use_cache)
        output = self.head(hidden, labels, aux_losses, new_caches)
        if return_hidden_states:
            output.hidden_states = hidden
        return output

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> ModelOutput:
        return self(input_ids, use_cache=True)

    def estimate_num_params(self, non_embedding: bool = False) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            count -= self.backbone.embedding.embed.weight.numel()
        return count


def NanoforgeForCausalLM(config: ModelConfig) -> NanoforgeModel:
    """Backward-compat factory: always returns a causal LM model."""
    from copy import copy
    cfg = copy(config)
    cfg.task_type = "causal_lm"
    cfg.is_causal = True
    return NanoforgeModel(cfg)


def build_model(config: NanoforgeConfig) -> NanoforgeModel:
    """Primary factory used by trainer and engine."""
    return NanoforgeModel(config.model)
