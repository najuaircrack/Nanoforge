from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F


@dataclass
class SamplingConfig:
    mode: Literal["balanced", "chat", "creative", "coding", "deterministic", "low_memory", "high_quality"] = "balanced"
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    min_p: float | None = None
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    deterministic: bool = False
    stop_on_repetition: bool = True
    repetition_window: int = 64
    repetition_threshold: float = 0.85
    mirostat: bool = False
    mirostat_version: int = 2
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    entropy_target: float | None = None
    adaptive_temperature: bool = False


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


def apply_frequency_presence_penalties(
    logits: torch.Tensor,
    history: torch.Tensor,
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    if history.numel() == 0 or (frequency_penalty == 0.0 and presence_penalty == 0.0):
        return logits
    logits = logits.clone()
    vocab = logits.shape[-1]
    counts = torch.bincount(history.clamp(0, vocab - 1), minlength=vocab).to(logits.device, logits.dtype)
    penalty = counts * frequency_penalty + (counts > 0).to(logits.dtype) * presence_penalty
    return logits - penalty.unsqueeze(0)


def block_repeated_ngrams(logits: torch.Tensor, history: torch.Tensor, ngram_size: int) -> torch.Tensor:
    if ngram_size <= 1 or history.numel() + 1 < ngram_size:
        return logits
    tokens = history.tolist()
    prefix = tuple(tokens[-(ngram_size - 1) :])
    banned: set[int] = set()
    for idx in range(len(tokens) - ngram_size + 1):
        gram = tuple(tokens[idx : idx + ngram_size - 1])
        if gram == prefix:
            banned.add(tokens[idx + ngram_size - 1])
    if not banned:
        return logits
    logits = logits.clone()
    valid = [token for token in banned if 0 <= token < logits.shape[-1]]
    if valid:
        logits[:, valid] = float("-inf")
    return logits


def min_p_filter(logits: torch.Tensor, min_p: float | None) -> torch.Tensor:
    if min_p is None or min_p <= 0:
        return logits
    probs = F.softmax(logits, dim=-1)
    threshold = probs.max(dim=-1, keepdim=True).values * min_p
    return logits.masked_fill(probs < threshold, float("-inf"))


def effective_sampling_config(config: SamplingConfig) -> SamplingConfig:
    tuned = SamplingConfig(**config.__dict__)
    if tuned.mode == "deterministic":
        tuned.temperature = 0.0
        tuned.top_k = 1
        tuned.top_p = None
        tuned.deterministic = True
    elif tuned.mode == "chat":
        tuned.temperature = min(tuned.temperature, 0.7)
        tuned.top_p = min(tuned.top_p or 0.9, 0.9)
        tuned.repetition_penalty = max(tuned.repetition_penalty, 1.08)
        tuned.no_repeat_ngram_size = max(tuned.no_repeat_ngram_size, 4)
    elif tuned.mode == "creative":
        tuned.temperature = max(tuned.temperature, 0.9)
        tuned.top_p = max(tuned.top_p or 0.95, 0.95)
    elif tuned.mode == "coding":
        tuned.temperature = min(tuned.temperature, 0.35)
        tuned.top_p = min(tuned.top_p or 0.9, 0.9)
        tuned.no_repeat_ngram_size = max(tuned.no_repeat_ngram_size, 5)
    elif tuned.mode == "low_memory":
        tuned.top_k = min(tuned.top_k or 40, 40)
    elif tuned.mode == "high_quality":
        tuned.top_p = min(tuned.top_p or 0.92, 0.92)
        tuned.repetition_penalty = max(tuned.repetition_penalty, 1.05)
    return tuned


def sample_next(
    logits: torch.Tensor,
    history: torch.Tensor,
    config: SamplingConfig,
    mirostat_state: MirostatState | None = None,
) -> torch.Tensor:
    config = effective_sampling_config(config)
    logits = logits[:, -1, :]
    logits = apply_repetition_penalty(logits, history, config.repetition_penalty)
    logits = apply_frequency_presence_penalties(logits, history, config.frequency_penalty, config.presence_penalty)
    logits = block_repeated_ngrams(logits, history, config.no_repeat_ngram_size)
    if config.deterministic or config.temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    temperature = max(config.temperature, 1e-5)
    if config.adaptive_temperature or config.entropy_target is not None:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
        target = config.entropy_target or config.mirostat_tau
        temperature *= float(torch.clamp(torch.tensor(target, device=logits.device) / entropy.clamp_min(1e-6), 0.5, 1.5).item())
    if config.mirostat and mirostat_state is not None:
        logits = logits / temperature
        logits = top_k_top_p_filter(logits, max(1, int(torch.exp(torch.tensor(mirostat_state.mu)).item())), None)
        logits = min_p_filter(logits, config.min_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        observed = -torch.log2(torch.gather(probs, -1, token).clamp_min(1e-12))
        mirostat_state.mu -= config.mirostat_eta * (observed.item() - config.mirostat_tau)
        return token
    logits = top_k_top_p_filter(logits / temperature, config.top_k, config.top_p)
    logits = min_p_filter(logits, config.min_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
