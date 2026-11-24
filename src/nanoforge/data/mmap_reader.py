"""Read-only bounded mmap corpus iteration with an optional Rust implementation.

Callers must supply an immutable file during iteration, as for NumPy memmaps.
Use the buffered dataset readers when a source may be modified concurrently.
"""
from __future__ import annotations

import mmap
from pathlib import Path

from nanoforge.data.native_tokenizer import load_native_module


class MmapCorpusReader:
    def __init__(self, path: str | Path, *, max_record_bytes: int = 1_048_576,
                 max_source_bytes: int = 268_435_456, require_native: bool = False,
                 prefer_native: bool = True):
        if min(max_record_bytes, max_source_bytes) < 1:
            raise ValueError("Mmap source and record limits must be positive")
        self.path = Path(path)
        self.max_record_bytes = max_record_bytes
        self._native = None
        self._file = None
        self._mapping = None
        self._position = 0
        self._records = self._bytes_read = self._invalid_utf8_records = self._skipped_records = 0
        module = load_native_module() if prefer_native or require_native else None
        native_type = getattr(module, "MmapCorpusReader", None) if module is not None else None
        if native_type is not None:
            self._native = native_type(str(self.path), max_record_bytes, max_source_bytes)
            return
        if require_native:
            raise RuntimeError("Native mmap reader is unavailable")
        self._file = self.path.open("rb")
        try:
            self._size = self.path.stat().st_size
            if self._size > max_source_bytes:
                raise ValueError("Mmap source byte limit exceeded")
            if self._size:
                self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self.close()
            raise

    @property
    def backend(self):
        return "rust-mmap" if self._native is not None else "python-mmap"

    @property
    def records(self):
        return self._native.records if self._native is not None else self._records

    @property
    def bytes_read(self):
        return self._native.bytes_read if self._native is not None else self._bytes_read

    @property
    def invalid_utf8_records(self):
        return self._native.invalid_utf8_records if self._native is not None else self._invalid_utf8_records

    @property
    def skipped_records(self):
        return self._native.skipped_records if self._native is not None else self._skipped_records

    def __iter__(self):
        return self

    def __next__(self):
        if self._native is not None:
            return next(self._native)
        if self._mapping is None or self._position >= self._size:
            raise StopIteration
        newline = self._mapping.find(b"\n", self._position)
        end = self._size if newline < 0 else newline + 1
        start = self._position
        self._position = end
        self._bytes_read += end - start
        if end - start > self.max_record_bytes:
            self._skipped_records += 1
            raise ValueError("Mmap record byte limit exceeded")
        data = self._mapping[start:end]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
            self._invalid_utf8_records += 1
        self._records += 1
        return text

    def close(self):
        if getattr(self, "_native", None) is not None:
            self._native.close()
        if getattr(self, "_mapping", None) is not None:
            self._mapping.close()
            self._mapping = None
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

