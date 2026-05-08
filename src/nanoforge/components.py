from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import nn


class AttentionBackend(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        cache: Any | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any | None]: ...


class FFNBackend(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


class PositionalEmbedding(Protocol):
    def get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class TokenizerBackend(Protocol):
    bos_id: int
    eos_id: int
    pad_id: int
    unk_id: int
    vocab_size: int

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class SamplerBackend(Protocol):
    def __call__(self, logits: torch.Tensor, history: torch.Tensor, config: Any) -> torch.Tensor: ...


class NormalizationBackend(nn.Module, Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class QuantizationBackend(Protocol):
    def quantize_module(self, module: nn.Module) -> nn.Module: ...
