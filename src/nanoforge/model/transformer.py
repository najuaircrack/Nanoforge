from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from nanoforge.config import ModelConfig
from nanoforge.model.attention import KVCache
from nanoforge.registry import ATTENTION_BACKENDS, FFN_BACKENDS, NORMALIZATIONS, TRANSFORMER_BLOCKS


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    caches: list[KVCache] | None = None


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        norm_factory = NORMALIZATIONS.get(config.normalization)
        self.attn_norm = norm_factory(config.d_model, config.norm_eps)
        self.ffn_norm = norm_factory(config.d_model, config.norm_eps)
        self.attn = ATTENTION_BACKENDS.create(config.attention_backend, config)
        hidden = int(config.d_model * config.ffn_hidden_mult)
        hidden = int(256 * math.ceil(hidden / 256)) if hidden >= 256 else hidden
        self.moe = config.moe is not None
        if self.moe:
            self.ffn = FFN_BACKENDS.create(
                "moe",
                config.d_model,
                hidden,
                config.moe,
                config.activation,
                config.dropout,
            )
        else:
            ffn_type = config.ffn_type or config.activation
            self.ffn = FFN_BACKENDS.create(
                ffn_type,
                config.d_model,
                hidden,
                ffn_type if ffn_type in {"swiglu", "geglu"} else config.activation,
                config.dropout,
            )
        self.residual_scale = config.residual_scale or (1.0 / math.sqrt(2 * config.n_layers))

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None, torch.Tensor | None]:
        attn_out, cache = self.attn(self.attn_norm(x), cache=cache, use_cache=use_cache)
        x = x + attn_out * self.residual_scale
        ffn_in = self.ffn_norm(x)
        aux_loss = None
        if self.moe:
            ffn_out, aux_loss = self.ffn(ffn_in)
        else:
            ffn_out = self.ffn(ffn_in)
        x = x + ffn_out * self.residual_scale
        return x, cache, aux_loss


class NanoforgeForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TRANSFORMER_BLOCKS.create(config.block_type, config) for _ in range(config.n_layers)]
        )
        self.norm = NORMALIZATIONS.create(config.normalization, config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight
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

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        if input_ids.shape[1] > self.config.max_seq_len and not use_cache:
            raise ValueError(f"Sequence length exceeds max_seq_len={self.config.max_seq_len}.")
        x = self.drop(self.embed(input_ids))
        new_caches: list[KVCache] = []
        aux_losses: list[torch.Tensor] = []
        if caches is None:
            caches = [None] * len(self.blocks)  # type: ignore[list-item]

        for block, cache in zip(self.blocks, caches):
            if self.config.gradient_checkpointing and self.training and not use_cache:
                def checkpoint_block(hidden: torch.Tensor) -> torch.Tensor:
                    y, _, _ = block(hidden, None, False)
                    return y

                x = torch.utils.checkpoint.checkpoint(checkpoint_block, x, use_reentrant=False)
                aux = None
                new_cache = None
            else:
                x, new_cache, aux = block(x, cache=cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(new_cache or KVCache())
            if aux is not None:
                aux_losses.append(aux)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        if loss is not None and aux_loss is not None and self.config.moe is not None:
            loss = loss + self.config.moe.router_aux_loss_coef * aux_loss
        return CausalLMOutput(logits=logits, loss=loss, aux_loss=aux_loss, caches=new_caches or None)

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> CausalLMOutput:
        return self(input_ids, use_cache=True)

    def estimate_num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n
