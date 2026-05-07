from __future__ import annotations

import math
from typing import Iterable

import torch


class Lion(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1.0 - lr * wd)
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class Adafactor(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta2: float = 0.999,
        eps: tuple[float, float] = (1e-30, 1e-3),
        weight_decay: float = 0.0,
    ):
        super().__init__(params, {"lr": lr, "beta2": beta2, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, beta2, eps, wd = group["lr"], group["beta2"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                if wd:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if grad.ndim >= 2:
                    if not state:
                        state["row"] = torch.zeros(grad.shape[:-1], device=grad.device)
                        state["col"] = torch.zeros(grad.shape[-1], device=grad.device)
                    row = state["row"]
                    col = state["col"]
                    grad_sq = grad.pow(2).add_(eps[0])
                    row.mul_(beta2).add_(grad_sq.mean(dim=-1), alpha=1 - beta2)
                    col.mul_(beta2).add_(grad_sq.mean(dim=tuple(range(grad.ndim - 1))), alpha=1 - beta2)
                    update = grad / (row.unsqueeze(-1).sqrt() * col.sqrt().clamp_min(eps[1]))
                else:
                    if not state:
                        state["v"] = torch.zeros_like(grad)
                    v = state["v"]
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    update = grad / v.sqrt().clamp_min(eps[1])
                rms = update.pow(2).mean().sqrt().clamp_min(1.0)
                p.add_(update.to(p.dtype), alpha=-lr / float(rms))
        return loss


class SophiaG(torch.optim.Optimizer):
    """Small Sophia-G style optimizer scaffold with diagonal Hessian EMA."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, {"lr": lr, "betas": betas, "rho": rho, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, betas, rho, wd = group["lr"], group["betas"], group["rho"], group["weight_decay"]
            beta1, beta2 = betas
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if not state:
                    state["m"] = torch.zeros_like(p)
                    state["h"] = torch.zeros_like(p)
                m, h = state["m"], state["h"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                h.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = (m / (rho * h.clamp_min(1e-12))).clamp_(-1, 1)
                p.add_(update, alpha=-lr)
        return loss


def create_optimizer(
    name: str,
    params,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
):
    lowered = name.lower()
    if lowered == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=torch.cuda.is_available(),
        )
    if lowered == "lion":
        return Lion(params, lr=lr, betas=(betas[0], 0.99), weight_decay=weight_decay)
    if lowered == "adafactor":
        return Adafactor(params, lr=lr, weight_decay=weight_decay)
    if lowered in {"sophia", "sophiag"}:
        return SophiaG(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")
