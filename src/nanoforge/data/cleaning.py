from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Iterator

from nanoforge.data.formats import DatasetRecord
from nanoforge.data.preprocess import stable_hash


@dataclass
class CleaningConfig:
    min_chars: int = 16
    max_chars: int | None = None
    normalize_unicode: bool = True
    collapse_whitespace: bool = False
    deduplicate: bool = True
    near_deduplicate: bool = False
    language: str | None = None
    near_window: int = 50000
    near_hamming: int = 3
    preserve_code: bool = True
    max_batch_bytes: int = 4_194_304

    def __post_init__(self):
        if self.min_chars < 0 or (self.max_chars is not None and self.max_chars < 1):
            raise ValueError("Invalid cleaning character limits")
        if self.near_window < 1 or not 0 <= self.near_hamming <= 64 or self.max_batch_bytes < 1:
            raise ValueError("Invalid near-duplicate limits")


def clean_text(text: str, config: CleaningConfig) -> str:
    text = text.replace("\x00", "")
    if config.normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if config.collapse_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def likely_language(text: str) -> str:
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    if ascii_count / max(len(text), 1) > 0.85:
        return "en"
    return "unknown"


def simhash(text: str, bits: int = 64) -> int:
    weights = [0] * bits
    tokens = re.findall(r"\w+", text.lower())
    for token in tokens:
        value = 0xcbf29ce484222325
        for byte in token.encode("utf-8"):
            value = ((value ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
        for bit in range(bits):
            weights[bit] += 1 if value & (1 << bit) else -1
    out = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _load_native_cleaning():
    try:
        import nanoforge_tokenizers as _m
        if hasattr(_m, "StreamingDeduplicator"):
            return _m
    except Exception:
        pass
    return None


@dataclass
class CleaningStats:
    seen: int = 0
    kept: int = 0
    too_short: int = 0
    truncated: int = 0
    language_rejected: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0


def clean_records(records: Iterable[DatasetRecord], config: CleaningConfig,
                  *, stats: CleaningStats | None = None) -> Iterator[DatasetRecord]:
    """One common filtering contract, optionally accelerating normalization only."""
    import sqlite3
    import tempfile
    from collections import deque
    from pathlib import Path

    stats = stats if stats is not None else CleaningStats()
    native = _load_native_cleaning()
    normalizer = getattr(native, "clean_text_batch", None) if native is not None else None

    def normalized():
        def is_code(record):
            return config.preserve_code and (record.metadata.get("mode") in {"code", "fim"} or record.metadata.get("format") == "code" or Path(record.source).suffix.lower() in {".py", ".c", ".cpp", ".h", ".hpp", ".rs", ".js", ".jsx", ".ts", ".tsx", ".java", ".go"})

        def normalize_record(record):
            if is_code(record):
                return record.text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
            return clean_text(record.text, config)

        if not callable(normalizer):
            for record in records:
                yield record, normalize_record(record)
            return

        def flush(batch):
            plain = [item for item in batch if not is_code(item)]
            output = normalizer([item.text for item in plain], config.normalize_unicode,
                                config.collapse_whitespace) if plain else []
            if len(output) != len(plain):
                raise ValueError("Native cleaning result count differs from inputs")
            result = iter(output)
            for item in batch:
                yield item, normalize_record(item) if is_code(item) else next(result)[0]

        batch = []
        batch_bytes = 0
        for record in records:
            size = len(record.text.encode("utf-8"))
            if batch and batch_bytes + size > config.max_batch_bytes:
                yield from flush(batch)
                batch = []
                batch_bytes = 0
            batch.append(record)
            batch_bytes += size
            if len(batch) == 256:
                yield from flush(batch)
                batch = []
                batch_bytes = 0
        if batch:
            yield from flush(batch)

    with tempfile.TemporaryDirectory(prefix="nanoforge-dedup-") as directory:
        connection = sqlite3.connect(str(Path(directory) / "seen.sqlite"))
        connection.execute("create table seen (hash text primary key)")
        connection.execute("pragma cache_size=-2048")
        seen_near = deque(maxlen=config.near_window)
        try:
            for record, text in normalized():
                stats.seen += 1
                if len(text) < config.min_chars:
                    stats.too_short += 1
                    continue
                if config.max_chars is not None and len(text) > config.max_chars:
                    text = text[:config.max_chars].rstrip() + "\n"
                    stats.truncated += 1
                if config.language is not None and likely_language(text) != config.language:
                    stats.language_rejected += 1
                    continue
                if config.deduplicate:
                    key = stable_hash(text)
                    cursor = connection.execute("insert or ignore into seen values (?)", (key,))
                    if not cursor.rowcount:
                        stats.exact_duplicates += 1
                        continue
                if config.near_deduplicate:
                    fingerprint = simhash(text)
                    if any(hamming(fingerprint, old) <= config.near_hamming for old in seen_near):
                        stats.near_duplicates += 1
                        continue
                    seen_near.append(fingerprint)
                record.text = text
                stats.kept += 1
                yield record
        finally:
            connection.close()


def _clean_records_python(records: Iterable[DatasetRecord], config: CleaningConfig) -> Iterator[DatasetRecord]:
    # Backward-compatible private helper; filtering is always the canonical path.
    yield from clean_records(records, config)


def _clean_records_native(records: Iterable[DatasetRecord], config: CleaningConfig, native) -> Iterator[DatasetRecord]:
    yield from clean_records(records, config)

