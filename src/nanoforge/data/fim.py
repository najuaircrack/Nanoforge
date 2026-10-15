"""Fill-in-the-middle (FIM) formatting for coding data.

FIM is a training objective used by modern coding models. It is not a new
model backbone: it changes the sequence format so the model learns to generate
the missing middle from a prefix and suffix. The tokenizer used for real
training must preserve the configured sentinel tokens as dedicated tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FIMConfig:
    prefix_token: str = "<|fim_prefix|>"
    suffix_token: str = "<|fim_suffix|>"
    middle_token: str = "<|fim_middle|>"
    split_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.split_fraction < 1.0:
            raise ValueError("split_fraction must be between 0 and 1")


def transform_fim_text(text: str, config: FIMConfig | None = None) -> str:
    """Format one code sample as prefix/suffix/middle deterministically."""

    config = config or FIMConfig()
    if not text:
        return f"{config.prefix_token}{config.suffix_token}{config.middle_token}"
    split_at = max(1, min(len(text) - 1, int(len(text) * config.split_fraction)))
    prefix = text[:split_at]
    middle = text[split_at:]
    return f"{config.prefix_token}{prefix}{config.suffix_token}{middle}{config.middle_token}"
