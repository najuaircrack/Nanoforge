from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from nanoforge.config import ModelConfig
    from nanoforge.model.attention import KVCache


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    caches: list[KVCache] | None = None
    hidden_states: torch.Tensor | None = None
    pooled_output: torch.Tensor | None = None


# Backward-compat alias
CausalLMOutput = ModelOutput


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CausalLMHead(nn.Module):
    def __init__(self, config: ModelConfig, tie_weight: torch.Tensor | None = None):
        super().__init__()
        self.proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if tie_weight is not None:
            self.proj.weight = tie_weight
        else:
            _init_linear(self.proj)
        self._moe_cfg = config.moe

    def forward(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
        aux_losses: list[torch.Tensor],
        caches: list[KVCache] | None,
    ) -> ModelOutput:
        logits = self.proj(hidden)
        loss = None
        if labels is not None:
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)
            n_tokens = (flat_labels != -100).sum()
            if n_tokens == 0:
                loss = flat_logits.sum() * 0.0
            else:
                loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum") / n_tokens
        aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        if loss is not None and aux_loss is not None and self._moe_cfg is not None:
            loss = loss + self._moe_cfg.router_aux_loss_coef * aux_loss
        return ModelOutput(logits=logits, loss=loss, aux_loss=aux_loss, caches=caches or None)


class MaskedLMHead(nn.Module):
    """BERT-style masked language modelling head."""

    def __init__(self, config: ModelConfig, tie_weight: torch.Tensor | None = None):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        self.proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if tie_weight is not None:
            self.proj.weight = tie_weight
        self.apply(_init_linear)

    def forward(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
        aux_losses: list[torch.Tensor],
        caches: list[KVCache] | None,
    ) -> ModelOutput:
        x = F.gelu(self.dense(hidden))
        x = self.norm(x)
        logits = self.proj(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return ModelOutput(logits=logits, loss=loss, caches=None)


class SequenceClassificationHead(nn.Module):
    """Mean-pool hidden states → linear classifier."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.d_model, config.num_labels)
        self.num_labels = config.num_labels
        self.apply(_init_linear)

    def forward(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
        aux_losses: list[torch.Tensor],
        caches: list[KVCache] | None,
    ) -> ModelOutput:
        pooled = hidden.mean(dim=1)
        pooled = self.dropout(F.tanh(self.dense(pooled)))
        logits = self.proj(pooled)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(-1), labels.float())
            else:
                loss = F.cross_entropy(logits, labels)
        return ModelOutput(logits=logits, loss=loss, pooled_output=pooled, caches=None)


class TokenClassificationHead(nn.Module):
    """Per-token linear classifier (NER, POS tagging, etc.)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.d_model, config.num_labels)
        self.apply(_init_linear)

    def forward(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
        aux_losses: list[torch.Tensor],
        caches: list[KVCache] | None,
    ) -> ModelOutput:
        x = self.dropout(hidden)
        logits = self.proj(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return ModelOutput(logits=logits, loss=loss, caches=None)


class RewardModelHead(nn.Module):
    """Scalar reward / preference head. Returns a single score per sequence."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(config.d_model, 1, bias=False)
        _init_linear(self.proj)

    def forward(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor | None,
        aux_losses: list[torch.Tensor],
        caches: list[KVCache] | None,
    ) -> ModelOutput:
        pooled = hidden[:, -1, :]  # last token (EOS) as summary
        logits = self.proj(pooled)
        loss = None
        if labels is not None:
            # labels shape: (B,) float rewards or (B, 2) chosen/rejected pairs
            if labels.dim() == 2:
                chosen = logits[labels[:, 0].long()]
                rejected = logits[labels[:, 1].long()]
                loss = -F.logsigmoid(chosen - rejected).mean()
            else:
                loss = F.mse_loss(logits.squeeze(-1), labels.float())
        return ModelOutput(logits=logits, loss=loss, pooled_output=pooled, caches=None)


# Alias — same architecture, different semantic role
ValueHead = RewardModelHead
