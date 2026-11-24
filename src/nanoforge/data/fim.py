"""Fill-in-the-middle (FIM) formatting for coding data.

FIM is a training objective used by modern coding models. It is not a new
model backbone: it changes the sequence format so the model learns to generate
the missing middle from a prefix and suffix. The tokenizer used for real
training must preserve the configured sentinel tokens as dedicated tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanoforge.data.tokenizer import TokenizerLike


@dataclass(frozen=True)
class FIMConfig:
    prefix_token: str = "<|fim_prefix|>"
    suffix_token: str = "<|fim_suffix|>"
    middle_token: str = "<|fim_middle|>"
    split_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.split_fraction < 1.0:
            raise ValueError("split_fraction must be between 0 and 1")


def validate_fim_tokenizer(tokenizer: TokenizerLike, config: FIMConfig | None = None) -> None:
    """Require distinct sentinels for quality paths; bytes remain smoke-compatible."""
    from nanoforge.data.native_tokenizer import NativeByteTokenizer
    from nanoforge.data.tokenizer import ByteTokenizer

    if isinstance(tokenizer, (ByteTokenizer, NativeByteTokenizer)):
        return
    config = config or FIMConfig()
    sentinels = [tokenizer.encode(token) for token in (
        config.prefix_token, config.suffix_token, config.middle_token,
    )]
    reserved = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id}
    if any(len(ids) != 1 or ids[0] in reserved for ids in sentinels) or len({tuple(ids) for ids in sentinels}) != 3:
        raise ValueError(
            "Real FIM requires distinct dedicated canonical sentinel tokens; "
            "legacy tokenizers require explicit migration and retraining/adaptation"
        )


def transform_fim_text(text: str, config: FIMConfig | None = None) -> str:
    """Format one code sample as prefix/suffix/middle deterministically."""

    config = config or FIMConfig()
    if not text:
        return f"{config.prefix_token}{config.suffix_token}{config.middle_token}"
    split_at = max(1, min(len(text) - 1, int(len(text) * config.split_fraction)))
    prefix = text[:split_at]
    remainder = text[split_at:]
    middle_length = max(1, len(remainder) // 2)
    middle = remainder[:middle_length]
    suffix = remainder[middle_length:]
    return f"{config.prefix_token}{prefix}{config.suffix_token}{suffix}{config.middle_token}{middle}"
