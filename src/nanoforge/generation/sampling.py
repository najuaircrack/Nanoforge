from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F


@dataclass
class SamplingConfig:
    mode: Literal["balanced", "chat", "creative", "coding", "deterministic", "low_memory", "high_quality"] = "balanced"
    sampler: str = "top_k_top_p"
    num_beams: int = 4
    length_penalty: float = 0.8
    contrastive_alpha: float = 0.6
    contrastive_top_k: int = 4
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

    def __post_init__(self):
        for name in ("temperature", "length_penalty", "contrastive_alpha", "repetition_penalty",
                     "frequency_penalty", "presence_penalty", "repetition_threshold",
                     "mirostat_tau", "mirostat_eta", "top_p", "min_p", "entropy_target"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite number.")
        for name in ("num_beams", "contrastive_top_k", "no_repeat_ngram_size",
                     "repetition_window", "mirostat_version", "top_k"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{name} must be an integer.")
        for name in ("deterministic", "stop_on_repetition", "mirostat", "adaptive_temperature"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if not 0 <= self.length_penalty <= 2 or (self.entropy_target is not None and self.entropy_target <= 0):
            raise ValueError("Invalid length penalty or entropy target.")
        if self.mode not in {"balanced", "chat", "creative", "coding", "deterministic",
                             "low_memory", "high_quality"}:
            raise ValueError("Unknown sampling mode.")
        if self.sampler not in {"top_k_top_p", "mirostat", "beam", "contrastive"}:
            raise ValueError("Unknown sampler.")
        if self.temperature < 0 or not math.isfinite(self.temperature):
            raise ValueError("Sampling temperature must be finite and nonnegative.")
        for name in ("top_p", "min_p"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 < value <= 1):
                raise ValueError(f"{name} must be in (0, 1].")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be nonnegative.")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("Repetition penalty must be positive.")
        if self.no_repeat_ngram_size < 0 or self.repetition_window < 1:
            raise ValueError("Invalid repetition budget.")
        if not 0 <= self.repetition_threshold <= 1:
            raise ValueError("Repetition threshold must be in [0, 1].")
        if self.mirostat_version not in (1, 2) or self.mirostat_tau <= 0 or self.mirostat_eta <= 0:
            raise ValueError("Invalid Mirostat version or parameters.")
        if not 0 <= self.contrastive_alpha <= 1 or self.contrastive_top_k < 1 or self.num_beams < 1:
            raise ValueError("Invalid contrastive/beam parameters.")
        if self.sampler == "mirostat":
            self.mirostat = True


class MirostatState:
    def __init__(self, tau: float):
        self.mu = 2.0 * tau


def apply_repetition_penalty(logits: torch.Tensor, history: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or history.numel() == 0:
        return logits
    if torch.any(history < 0) or torch.any(history >= logits.shape[-1]):
        raise ValueError("Sampling history contains invalid vocabulary IDs.")
    unique = history.unique()
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
    if ngram_size < 1 or history.numel() + 1 < ngram_size:
        return logits
    if ngram_size == 1:
        logits = logits.clone()
        logits[:, history.unique()] = -torch.inf
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
    if logits.ndim != 3 or logits.shape[1] == 0:
        raise ValueError("Sampling requires nonempty [batch, sequence, vocabulary] logits.")
    logits = logits[:, -1, :]
    if history.ndim != 1 or history.dtype not in (torch.int32, torch.int64):
        raise ValueError("Sampling history must be a one-dimensional integer tensor.")
    if torch.any(history < 0) or torch.any(history >= logits.shape[-1]):
        raise ValueError("Sampling history contains invalid vocabulary IDs.")
    if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError("Sampling logits contain NaNs, positive infinity, or no valid continuation.")
    logits = apply_repetition_penalty(logits, history, config.repetition_penalty)
    logits = apply_frequency_presence_penalties(logits, history, config.frequency_penalty, config.presence_penalty)
    logits = block_repeated_ngrams(logits, history, config.no_repeat_ngram_size)
    if not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError("Sampling penalties removed every valid continuation.")
    if config.deterministic or config.temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    temperature = max(config.temperature, 1e-5)
    if config.adaptive_temperature or config.entropy_target is not None:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
        target = config.entropy_target or config.mirostat_tau
        temperature *= float(torch.clamp(torch.tensor(target, device=logits.device) / entropy.clamp_min(1e-6), 0.5, 1.5).item())
    if config.mirostat and mirostat_state is not None:
        if logits.shape[0] != 1:
            raise ValueError("Mirostat requires an independent state per sequence.")
        scaled = logits / temperature
        full_probs = F.softmax(scaled, dim=-1)
        if config.mirostat_version == 2:
            surprise = -torch.log2(full_probs.clamp_min(1e-30))
            cutoff = surprise > mirostat_state.mu
            cutoff.scatter_(-1, full_probs.argmax(-1, keepdim=True), False)
            scaled = scaled.masked_fill(cutoff, -torch.inf)
        else:
            sorted_probs = full_probs.sort(descending=True, dim=-1).values[0]
            m = min(100, sorted_probs.numel() - 1)
            if m < 1:
                k = 1
            else:
                rank = torch.arange(1, m + 1, device=logits.device, dtype=torch.float32)
                t = torch.log((rank + 1) / rank)
                b = torch.log(sorted_probs[:m].clamp_min(1e-30) /
                              sorted_probs[1:m + 1].clamp_min(1e-30))
                slope = float((t * b).sum() / t.square().sum())
                slope = max(slope, 1e-3)
                epsilon = slope - 1
                normalization = (-math.expm1(-epsilon * math.log(sorted_probs.numel()))
                                 if abs(epsilon) > 1e-6 else math.log(sorted_probs.numel()))
                ratio = epsilon / normalization if abs(epsilon) > 1e-6 else 1 / normalization
                log_k = (math.log(max(ratio, 1e-30)) + min(mirostat_state.mu, 64) * math.log(2)) / slope
                k = max(1, min(sorted_probs.numel(), int(math.exp(min(log_k, math.log(sorted_probs.numel()))))))
            scaled = top_k_top_p_filter(scaled, k, None)
        scaled = min_p_filter(scaled, config.min_p)
        probs = F.softmax(scaled, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        observed = -torch.log2(torch.gather(probs, -1, token).clamp_min(1e-30))
        mirostat_state.mu -= config.mirostat_eta * (observed.item() - config.mirostat_tau)
        return token
    logits = top_k_top_p_filter(logits / temperature, config.top_k, config.top_p)
    logits = min_p_filter(logits, config.min_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def sampling_probabilities(logits, history, config):
    """The normalized stateless sampling distribution used by speculative verification."""
    config = effective_sampling_config(config)
    if config.mirostat or config.sampler in {"beam", "contrastive"}:
        raise ValueError("Speculative distributions require a stateless sampling configuration.")
    if logits.ndim == 3:
        logits = logits[:, -1]
    logits = apply_repetition_penalty(logits, history, config.repetition_penalty)
    logits = apply_frequency_presence_penalties(logits, history, config.frequency_penalty, config.presence_penalty)
    logits = block_repeated_ngrams(logits, history, config.no_repeat_ngram_size)
    if (not torch.isfinite(logits).any(-1).all() or torch.isnan(logits).any()
            or torch.isposinf(logits).any()):
        raise ValueError("Speculative distribution has no valid continuation.")
    if config.deterministic or config.temperature <= 0:
        return torch.zeros_like(logits).scatter_(-1, logits.argmax(-1, keepdim=True), 1.)
    temperature = config.temperature
    if config.adaptive_temperature or config.entropy_target is not None:
        probabilities = F.softmax(logits, -1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
        target = config.entropy_target or config.mirostat_tau
        temperature *= float((target / entropy.clamp_min(1e-6)).clamp(.5, 1.5))
    logits = top_k_top_p_filter(logits / temperature, config.top_k, config.top_p)
    logits = min_p_filter(logits, config.min_p)
    return F.softmax(logits, -1)


def top_k_top_p_generate(engine, input_ids, max_new_tokens, sampling=None, **kwargs):
    from dataclasses import replace

    config = replace(sampling or SamplingConfig(), sampler="top_k_top_p", mirostat=False)
    yield from engine.generate_ids(input_ids, max_new_tokens, config, **kwargs)


def mirostat_generate(engine, input_ids, max_new_tokens, sampling=None, **kwargs):
    from dataclasses import replace

    config = replace(sampling or SamplingConfig(), sampler="mirostat", mirostat=True)
    yield from engine.generate_ids(input_ids, max_new_tokens, config, **kwargs)


def beam_generate(engine, input_ids, max_new_tokens, sampling=None, **kwargs):
    from dataclasses import replace

    config = replace(sampling or SamplingConfig(), sampler="beam", mirostat=False)
    yield from engine.generate_ids(input_ids, max_new_tokens, config, **kwargs)


def contrastive_generate(engine, input_ids, max_new_tokens, sampling=None, **kwargs):
    from dataclasses import replace

    config = replace(sampling or SamplingConfig(), sampler="contrastive", mirostat=False)
    yield from engine.generate_ids(input_ids, max_new_tokens, config, **kwargs)

