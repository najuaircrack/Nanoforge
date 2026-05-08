from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerConfig:
    max_steps: int
    warmup_steps: int
    learning_rate: float
    min_learning_rate: float = 0.0


class LearningRateScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config

    def __call__(self, step: int) -> float:
        raise NotImplementedError

    def _warmup(self, step: int) -> float | None:
        if self.config.warmup_steps > 0 and step < self.config.warmup_steps:
            return self.config.learning_rate * (step + 1) / self.config.warmup_steps
        return None


class CosineScheduler(LearningRateScheduler):
    def __call__(self, step: int) -> float:
        warmup = self._warmup(step)
        if warmup is not None:
            return warmup
        progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return self.config.min_learning_rate + coeff * (self.config.learning_rate - self.config.min_learning_rate)


class LinearScheduler(LearningRateScheduler):
    def __call__(self, step: int) -> float:
        warmup = self._warmup(step)
        if warmup is not None:
            return warmup
        progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
        coeff = max(0.0, 1.0 - progress)
        return self.config.min_learning_rate + coeff * (self.config.learning_rate - self.config.min_learning_rate)


class ConstantScheduler(LearningRateScheduler):
    def __call__(self, step: int) -> float:
        warmup = self._warmup(step)
        return self.config.learning_rate if warmup is None else warmup


def create_scheduler(
    name: str,
    *,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float = 0.0,
) -> LearningRateScheduler:
    config = SchedulerConfig(max_steps, warmup_steps, learning_rate, min_learning_rate)
    key = name.strip().lower().replace("-", "_")
    if key == "cosine":
        return CosineScheduler(config)
    if key == "linear":
        return LinearScheduler(config)
    if key == "constant":
        return ConstantScheduler(config)
    raise ValueError(f"Unknown scheduler '{name}'.")


def create_cosine_scheduler(**kwargs) -> CosineScheduler:
    return CosineScheduler(SchedulerConfig(**kwargs))


def create_linear_scheduler(**kwargs) -> LinearScheduler:
    return LinearScheduler(SchedulerConfig(**kwargs))


def create_constant_scheduler(**kwargs) -> ConstantScheduler:
    return ConstantScheduler(SchedulerConfig(**kwargs))
