from __future__ import annotations

import copy
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


def merge_lora(module: nn.Module, *, inplace: bool = False) -> nn.Module:
    """Merge adapters into dense Linear copies, preserving the source by default."""
    target = module if inplace else copy.deepcopy(module)
    if isinstance(target, LoRALinear):
        linear = nn.Linear(
            target.weight.shape[1],
            target.weight.shape[0],
            bias=target.bias is not None,
            device=target.weight.device,
            dtype=target.weight.dtype,
        )
        with torch.no_grad():
            weight = target.weight
            if target.rank:
                weight = weight + (target.lora_b @ target.lora_a) * target.scaling
            linear.weight.copy_(weight)
            if target.bias is not None:
                linear.bias.copy_(target.bias)
        linear.train(target.training)
        return linear
    for name, child in list(target.named_children()):
        setattr(target, name, merge_lora(child, inplace=True))
    return target
