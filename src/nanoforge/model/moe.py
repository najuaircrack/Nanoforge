from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from nanoforge.config import MoEConfig


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str = "swiglu", dropout: float = 0.0):
        super().__init__()
        self.activation = activation
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "geglu":
            h = F.gelu(self.w1(x)) * self.w3(x)
        else:
            h = F.silu(self.w1(x)) * self.w3(x)
        return self.w2(self.dropout(h))


class MambaSelectiveStateFeedForward(nn.Module):
    """Lightweight selective-state style channel mixer.

    This is not a full Mamba SSM kernel; it provides a real gated sequence-mixing backend with
    the same FFN call surface for experiments on small models.
    """

    def __init__(self, dim: int, hidden_dim: int, activation: str = "swiglu", dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, groups=hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        mixed = self.depthwise(value.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return self.out_proj(self.dropout(F.silu(mixed) * torch.sigmoid(gate)))


class RWKVMixingFeedForward(nn.Module):
    """RWKV-inspired time-mixing FFN backend for experimentation."""

    def __init__(self, dim: int, hidden_dim: int, activation: str = "swiglu", dropout: float = 0.0):
        super().__init__()
        self.time_mix = nn.Parameter(torch.full((1, 1, dim), 0.5))
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        mixed = x * self.time_mix + shifted * (1.0 - self.time_mix)
        return torch.sigmoid(self.receptance(x)) * self.value(self.dropout(F.silu(self.key(mixed))))


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        config: MoEConfig,
        activation: str = "swiglu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.config = config
        self.router = nn.Linear(dim, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(dim, hidden_dim, activation, dropout) for _ in range(config.num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits.float(), dim=-1).to(x.dtype)
        weights, indices = router_probs.topk(self.config.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            mask = indices == expert_id
            if not mask.any():
                continue
            token_idx, rank_idx = mask.nonzero(as_tuple=True)
            expert_out = expert(flat[token_idx])
            out[token_idx] += expert_out * weights[token_idx, rank_idx].unsqueeze(-1)
        density = router_probs.mean(dim=0)
        density_proxy = (router_probs > (1.0 / self.config.num_experts)).float().mean(dim=0)
        aux_loss = (density * density_proxy).sum() * (self.config.num_experts**2)
        return out.reshape(shape), aux_loss
