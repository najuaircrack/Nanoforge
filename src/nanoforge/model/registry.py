from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        self._items[name] = factory

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            available = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown {self.name} '{name}'. Available: {available}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


ATTENTION_BACKENDS = Registry("attention backend")
FFN_BACKENDS = Registry("ffn backend")
ACTIVATIONS = Registry("activation")


def register_builtin_architecture_names() -> None:
    # These names are intentionally stable config targets. Some route to existing
    # implementations today, while experimental kernels can replace factories later.
    for name in [
        "sdpa",
        "flash",
        "flash_v2",
        "flash_v3",
        "manual",
        "sliding_window",
        "dilated",
        "sparse",
        "linear",
        "gated",
        "dynamic",
    ]:
        ATTENTION_BACKENDS.register(name, lambda *args, **kwargs: None)
    for name in ["swiglu", "geglu", "moe", "rwkv_mixing", "mamba_selective_state"]:
        FFN_BACKENDS.register(name, lambda *args, **kwargs: None)
    for name in ["silu", "gelu", "relu", "squared_relu", "swiglu", "geglu"]:
        ACTIVATIONS.register(name, lambda *args, **kwargs: None)


register_builtin_architecture_names()

