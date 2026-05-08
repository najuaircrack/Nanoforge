from __future__ import annotations

import torch
from torch.nn import functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)


def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return F.silu(x) * gate


def geglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return F.gelu(x) * gate
