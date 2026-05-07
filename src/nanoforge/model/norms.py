from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        scale = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * scale).to(dtype) * self.weight

