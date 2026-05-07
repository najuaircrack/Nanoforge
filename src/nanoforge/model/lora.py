from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Linear layer with optional low-rank trainable adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        rank: int = 0,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.dropout = nn.Dropout(dropout)
        if rank > 0:
            self.lora_a = nn.Parameter(torch.empty(rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            out = out + F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return out

    def freeze_base(self) -> None:
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)


def mark_only_lora_trainable(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = "lora_" in name

