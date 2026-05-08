from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import import_module, metadata
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    name: str
    target: Callable[..., T] | str
    version: str = "1"
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> Callable[..., T]:
        if callable(self.target):
            return self.target
        module_name, _, attr = self.target.partition(":")
        if not module_name or not attr:
            raise ValueError(f"Registry target for '{self.name}' must be 'module:attribute'.")
        module = import_module(module_name)
        factory = getattr(module, attr)
        if not callable(factory):
            raise TypeError(f"Registry target '{self.target}' is not callable.")
        return factory


class Registry(Generic[T]):
    """String-keyed component registry with lazy imports and metadata.

    Registries are intentionally small and dependency-free: component modules can be loaded only
    when a config selects them, while config validation can still list available keys.
    """

    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, RegistryEntry[T]] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        name: str,
        target: Callable[..., T] | str | None = None,
        *,
        aliases: Iterable[str] = (),
        version: str = "1",
        replace: bool = False,
        **metadata_values: Any,
    ):
        def decorator(factory: Callable[..., T] | str) -> Callable[..., T] | str:
            self._register_entry(
                RegistryEntry(
                    name=name,
                    target=factory,
                    aliases=tuple(aliases),
                    version=version,
                    metadata=dict(metadata_values),
                ),
                replace=replace,
            )
            return factory

        if target is None:
            return decorator
        return decorator(target)

    def _register_entry(self, entry: RegistryEntry[T], *, replace: bool = False) -> None:
        key = self._normalize(entry.name)
        if key in self._entries and not replace:
            raise KeyError(f"{self.name} '{entry.name}' is already registered.")
        self._entries[key] = entry
        for alias in entry.aliases:
            alias_key = self._normalize(alias)
            if alias_key in self._aliases and not replace:
                raise KeyError(f"{self.name} alias '{alias}' is already registered.")
            self._aliases[alias_key] = key

    def get(self, name: str) -> Callable[..., T]:
        ensure_registry_ready()
        return self.entry(name).resolve()

    def entry(self, name: str) -> RegistryEntry[T]:
        key = self._resolve_key(name)
        if key not in self._entries:
            available = ", ".join(self.names())
            raise KeyError(f"Unknown {self.name} '{name}'. Available: {available}")
        return self._entries[key]

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        return self.get(name)(*args, **kwargs)

    def contains(self, name: str) -> bool:
        return self._resolve_key(name) in self._entries

    def names(self) -> list[str]:
        return sorted(entry.name for entry in self._entries.values())

    def describe(self) -> dict[str, dict[str, Any]]:
        return {
            entry.name: {
                "version": entry.version,
                "aliases": list(entry.aliases),
                "metadata": entry.metadata,
            }
            for entry in self._entries.values()
        }

    def validate(self, name: str) -> None:
        self.entry(name)

    def _resolve_key(self, name: str) -> str:
        key = self._normalize(name)
        return self._aliases.get(key, key)

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower().replace("-", "_")


ATTENTION_BACKENDS: Registry[Any] = Registry("attention backend")
FFN_BACKENDS: Registry[Any] = Registry("ffn backend")
ACTIVATIONS: Registry[Any] = Registry("activation")
POSITION_EMBEDDINGS: Registry[Any] = Registry("positional embedding")
OPTIMIZERS: Registry[Any] = Registry("optimizer")
SCHEDULERS: Registry[Any] = Registry("scheduler")
TOKENIZERS: Registry[Any] = Registry("tokenizer")
SAMPLERS: Registry[Any] = Registry("sampler")
NORMALIZATIONS: Registry[Any] = Registry("normalization")
QUANTIZATION_BACKENDS: Registry[Any] = Registry("quantization backend")
TRANSFORMER_BLOCKS: Registry[Any] = Registry("transformer block")


ALL_REGISTRIES: dict[str, Registry[Any]] = {
    "attention": ATTENTION_BACKENDS,
    "ffn": FFN_BACKENDS,
    "activation": ACTIVATIONS,
    "position": POSITION_EMBEDDINGS,
    "optimizer": OPTIMIZERS,
    "scheduler": SCHEDULERS,
    "tokenizer": TOKENIZERS,
    "sampler": SAMPLERS,
    "normalization": NORMALIZATIONS,
    "quantization": QUANTIZATION_BACKENDS,
    "block": TRANSFORMER_BLOCKS,
}

_BUILTINS_REGISTERED = False
_PLUGINS_LOADED = False


def register_builtin_components() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    for name in ["sdpa", "flash", "flash_v2", "flash_v3", "manual", "sliding_window"]:
        ATTENTION_BACKENDS.register(
            name,
            "nanoforge.model.attention:CausalSelfAttention",
            version="1",
            replace=True,
            status="available",
        )
    ATTENTION_BACKENDS.register("chunked", "nanoforge.model.attention:ChunkedCausalSelfAttention", version="1", replace=True)
    ATTENTION_BACKENDS.register("sparse", "nanoforge.model.attention:SparseCausalSelfAttention", version="1", replace=True)
    ATTENTION_BACKENDS.register(
        "hybrid_local_global",
        "nanoforge.model.attention:HybridLocalGlobalCausalAttention",
        version="1",
        replace=True,
    )
    ATTENTION_BACKENDS.register("paged", "nanoforge.model.attention:PagedCausalSelfAttention", version="1", replace=True)

    FFN_BACKENDS.register("swiglu", "nanoforge.model.moe:FeedForward", version="1", replace=True)
    FFN_BACKENDS.register("geglu", "nanoforge.model.moe:FeedForward", version="1", replace=True)
    FFN_BACKENDS.register("moe", "nanoforge.model.moe:MoEFeedForward", version="1", replace=True)
    FFN_BACKENDS.register(
        "mamba_selective_state",
        "nanoforge.model.moe:MambaSelectiveStateFeedForward",
        version="1",
        replace=True,
    )
    FFN_BACKENDS.register(
        "rwkv_mixing",
        "nanoforge.model.moe:RWKVMixingFeedForward",
        version="1",
        replace=True,
    )

    NORMALIZATIONS.register("rmsnorm", "nanoforge.model.norms:RMSNorm", version="1", replace=True)
    NORMALIZATIONS.register("layernorm", "torch.nn:LayerNorm", version="1", replace=True)

    ACTIVATIONS.register("silu", "nanoforge.model.activations:silu", version="1", replace=True)
    ACTIVATIONS.register("gelu", "nanoforge.model.activations:gelu", version="1", replace=True)
    ACTIVATIONS.register("relu", "nanoforge.model.activations:relu", version="1", replace=True)
    ACTIVATIONS.register("squared_relu", "nanoforge.model.activations:squared_relu", version="1", replace=True)
    ACTIVATIONS.register("swiglu", "nanoforge.model.activations:swiglu", version="1", replace=True)
    ACTIVATIONS.register("geglu", "nanoforge.model.activations:geglu", version="1", replace=True)

    POSITION_EMBEDDINGS.register("rope", "nanoforge.model.rope:RotaryEmbedding", version="1", replace=True)
    POSITION_EMBEDDINGS.register("alibi", "nanoforge.model.attention:_alibi_bias", version="1", replace=True)
    POSITION_EMBEDDINGS.register("none", "nanoforge.model.positions:no_position_embedding", version="1", replace=True)

    for name in ["byte", "byte_native", "native_byte"]:
        TOKENIZERS.register(name, "nanoforge.data.tokenizer:load_tokenizer", version="1", replace=True)
    for name in ["bpe", "native_bpe", "python_bpe", "wordpiece", "sentencepiece", "unigram"]:
        TOKENIZERS.register(name, "nanoforge.data.tokenizer:load_tokenizer", version="1", replace=True)

    for name in ["top_k_top_p", "mirostat", "beam", "contrastive"]:
        SAMPLERS.register(name, "nanoforge.generation.sampling:sample_next", version="1", replace=True)

    OPTIMIZERS.register("adamw", "nanoforge.training.optimizers:create_adamw_optimizer", version="1", replace=True)
    OPTIMIZERS.register("lion", "nanoforge.training.optimizers:Lion", version="1", replace=True)
    OPTIMIZERS.register("adafactor", "nanoforge.training.optimizers:Adafactor", version="1", replace=True)
    OPTIMIZERS.register("sophia", "nanoforge.training.optimizers:SophiaG", version="1", replace=True)
    OPTIMIZERS.register("sophiag", "nanoforge.training.optimizers:SophiaG", version="1", replace=True)

    SCHEDULERS.register("cosine", "nanoforge.training.schedulers:create_cosine_scheduler", version="1", replace=True)
    SCHEDULERS.register("linear", "nanoforge.training.schedulers:create_linear_scheduler", version="1", replace=True)
    SCHEDULERS.register("constant", "nanoforge.training.schedulers:create_constant_scheduler", version="1", replace=True)

    for name in ["none", "int8", "int4", "gptq", "awq", "gguf"]:
        QUANTIZATION_BACKENDS.register(name, "nanoforge.quantization:apply_quantization", version="1", replace=True)

    TRANSFORMER_BLOCKS.register(
        "transformer",
        "nanoforge.model.transformer:TransformerBlock",
        version="1",
        replace=True,
    )
    TRANSFORMER_BLOCKS.register(
        "parallel_residual",
        "nanoforge.model.transformer:ParallelResidualTransformerBlock",
        version="1",
        replace=True,
    )


def load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    try:
        eps = metadata.entry_points()
        group = eps.select(group="nanoforge.plugins") if hasattr(eps, "select") else eps.get("nanoforge.plugins", [])
    except Exception:
        return
    for entry_point in group:
        plugin = entry_point.load()
        if callable(plugin):
            plugin(ALL_REGISTRIES)
        elif hasattr(plugin, "register"):
            plugin.register(ALL_REGISTRIES)


def ensure_registry_ready() -> None:
    register_builtin_components()
    load_plugins()


def validate_registry_key(registry_name: str, key: str) -> None:
    ensure_registry_ready()
    ALL_REGISTRIES[registry_name].validate(key)


def registry_snapshot() -> dict[str, dict[str, dict[str, Any]]]:
    ensure_registry_ready()
    return {name: registry.describe() for name, registry in ALL_REGISTRIES.items()}
