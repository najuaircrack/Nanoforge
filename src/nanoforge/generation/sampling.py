from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class SamplingConfig:
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.0
    mirostat: bool = False
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1


class MirostatState:
    def __init__(self, tau: float):
        self.mu = 2.0 * tau


def apply_repetition_penalty(logits: torch.Tensor, history: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or history.numel() == 0:
        return logits
    unique = history.unique().clamp_max(logits.shape[-1] - 1)
    gathered = logits[:, unique]
    adjusted = torch.where(gathered < 0, gathered * penalty, gathered / penalty)
    logits = logits.clone()
    logits[:, unique] = adjusted
    return logits


def top_k_top_p_filter(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    logits = logits.clone()
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


def sample_next(
    logits: torch.Tensor,
    history: torch.Tensor,
    config: SamplingConfig,
    mirostat_state: MirostatState | None = None,
) -> torch.Tensor:
    logits = logits[:, -1, :]
    logits = apply_repetition_penalty(logits, history, config.repetition_penalty)
    temperature = max(config.temperature, 1e-5)
    if config.mirostat and mirostat_state is not None:
        logits = logits / temperature
        logits = top_k_top_p_filter(logits, max(1, int(torch.exp(torch.tensor(mirostat_state.mu)).item())), None)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        observed = -torch.log2(torch.gather(probs, -1, token).clamp_min(1e-12))
        mirostat_state.mu -= config.mirostat_eta * (observed.item() - config.mirostat_tau)
        return token
    logits = top_k_top_p_filter(logits / temperature, config.top_k, config.top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
