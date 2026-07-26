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


class LayerNorm(nn.Module):
    """Standard LayerNorm with float32 reduction and optional bias.

    Mirrors ``RMSNorm``'s upcast-then-downcast contract so mixed-precision training
    stays numerically stable regardless of the input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = x_float.var(dim=-1, keepdim=True, unbiased=False)
        normed = (x_float - mean) * torch.rsqrt(var + self.eps)
        out = normed.to(dtype) * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

