from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class NativeTokenizerStatus:
    available: bool
    backend: str
    reason: str = ""


def load_native_module() -> Any | None:
    """Load the optional native tokenizer extension.

    Nanoforge keeps native acceleration optional so the Python package remains usable on
    machines without a Rust/C++ toolchain. The preferred extension module is the Rust
    `nanoforge_tokenizers` crate under `native/`.
    """

    for name in ("nanoforge_tokenizers", "nanoforge_native_tokenizers"):
        try:
            return import_module(name)
        except Exception:
            continue
    return None


def native_tokenizer_status() -> NativeTokenizerStatus:
    module = load_native_module()
    if module is None:
        return NativeTokenizerStatus(
            available=False,
            backend="python-fallback",
            reason="Native tokenizer extension is not installed.",
        )
    version = getattr(module, "__version__", "unknown")
    return NativeTokenizerStatus(available=True, backend=f"rust:{version}")


class NativeByteTokenizer:
    """Byte tokenizer facade with optional Rust acceleration.

    The ID layout intentionally matches `ByteTokenizer` for checkpoint and dataset compatibility.
    When the Rust extension is unavailable, the class falls back to the dependency-free Python
    implementation with identical behavior.
    """

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    offset = 4
    vocab_size = 260

    def __init__(self, *, require_native: bool = False):
        self._module = load_native_module()
        self._native = None
        if self._module is not None and hasattr(self._module, "ByteTokenizer"):
            self._native = self._module.ByteTokenizer()
        elif require_native:
            raise RuntimeError(
                "Native tokenizer extension is not installed. Build native/nanoforge-tokenizers "
                "or use tokenizer_type='byte' for the Python fallback."
            )

    @property
    def backend(self) -> str:
        return "rust" if self._native is not None else "python-fallback"

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if self._native is not None:
            return list(self._native.encode(text, add_bos, add_eos))
        ids = [byte + self.offset for byte in text.encode("utf-8", errors="replace")]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(
        self,
        texts: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        if self._native is not None and hasattr(self._native, "encode_batch"):
            return [list(ids) for ids in self._native.encode_batch(list(texts), add_bos, add_eos)]
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def decode(self, ids: list[int]) -> str:
        if self._native is not None:
            return str(self._native.decode(ids))
        data = bytes([idx - self.offset for idx in ids if self.offset <= idx < self.offset + 256])
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"type": "byte", "backend": self.backend, "native_optional": True}),
            encoding="utf-8",
        )


class NativeBPETokenizer:
    """Byte-level BPE facade with Rust acceleration and Python fallback."""

    def __init__(self, path: str | Path | None = None, *, require_native: bool = False):
        self.path = Path(path) if path is not None else None
        self._module = load_native_module()
        self._native = None
        self._fallback = None
        if self._module is not None and hasattr(self._module, "ByteLevelBpeTokenizer"):
            self._native = self._module.ByteLevelBpeTokenizer(str(self.path) if self.path is not None else None)
        elif require_native:
            raise RuntimeError(
                "Native BPE tokenizer extension is not installed. Build native/nanoforge-tokenizers "
                "or use tokenizer_type='python-bpe' for the Python fallback."
            )
        else:
            from nanoforge.data.tokenizer import PurePythonBPETokenizer

            self._fallback = PurePythonBPETokenizer(self.path)

    @property
    def backend(self) -> str:
        return "rust" if self._native is not None else "python-fallback"

    @property
    def pad_id(self) -> int:
        return int(self._native.pad_id if self._native is not None else self._fallback.pad_id)

    @property
    def bos_id(self) -> int:
        return int(self._native.bos_id if self._native is not None else self._fallback.bos_id)

    @property
    def eos_id(self) -> int:
        return int(self._native.eos_id if self._native is not None else self._fallback.eos_id)

    @property
    def unk_id(self) -> int:
        return int(self._native.unk_id if self._native is not None else self._fallback.unk_id)

    @property
    def vocab_size(self) -> int:
        return int(self._native.vocab_size if self._native is not None else self._fallback.vocab_size)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if self._native is not None:
            return list(self._native.encode(text, add_bos, add_eos))
        return self._fallback.encode(text, add_bos=add_bos, add_eos=add_eos)

    def encode_batch(
        self,
        texts: Iterable[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        if self._native is not None and hasattr(self._native, "encode_batch"):
            return [list(ids) for ids in self._native.encode_batch(list(texts), add_bos, add_eos)]
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def decode(self, ids: list[int]) -> str:
        if self._native is not None:
            return str(self._native.decode(ids))
        return self._fallback.decode(ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._native is not None:
            self._native.save(str(path))
        else:
            self._fallback.save(path)


def train_native_bpe_from_texts(
    texts: Iterable[str],
    out_path: str | Path,
    *,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    require_native: bool = False,
) -> bool:
    module = load_native_module()
    if module is None or not hasattr(module, "ByteLevelBpeTokenizer"):
        if require_native:
            raise RuntimeError("Native BPE tokenizer extension is not installed.")
        return False
    tokenizer = module.ByteLevelBpeTokenizer.train_from_texts(list(texts), vocab_size, min_frequency)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return True


def encode_batch(
    tokenizer: Any,
    texts: Iterable[str],
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[list[int]]:
    if hasattr(tokenizer, "encode_batch"):
        return tokenizer.encode_batch(texts, add_bos=add_bos, add_eos=add_eos)
    return [tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]
