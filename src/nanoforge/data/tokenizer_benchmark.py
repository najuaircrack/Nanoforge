from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nanoforge.data.formats import DatasetStats
from nanoforge.data.native_tokenizer import encode_batch, native_tokenizer_status
from nanoforge.data.tokenizer import TokenizerLike, iter_tokenizer_training_texts


@dataclass
class TokenizerBenchmarkResult:
    backend: str
    records: int
    chars: int
    tokens: int
    seconds: float
    chars_per_second: float
    tokens_per_second: float
    peak_memory_mb: float
    invalid_records: int
    skipped_records: int


def benchmark_tokenizer(
    tokenizer: TokenizerLike,
    paths: Iterable[str | Path],
    *,
    text_key: str = "text",
    text_columns: Iterable[str] | None = None,
    limit: int = 1000,
    batch_size: int = 64,
    add_bos: bool = False,
    add_eos: bool = False,
) -> TokenizerBenchmarkResult:
    stats = DatasetStats()
    backend = getattr(tokenizer, "backend", tokenizer.__class__.__name__)
    native = native_tokenizer_status()
    if backend == "python-fallback" and native.available:
        backend = native.backend

    records = 0
    chars = 0
    tokens = 0
    batch: list[str] = []
    start = time.perf_counter()
    tracemalloc.start()
    try:
        for text in iter_tokenizer_training_texts(
            paths,
            text_key=text_key,
            text_columns=text_columns,
            max_records=limit,
            stats=stats,
        ):
            batch.append(text)
            records += 1
            chars += len(text)
            if len(batch) >= batch_size:
                tokens += _count_batch_tokens(tokenizer, batch, add_bos=add_bos, add_eos=add_eos)
                batch.clear()
        if batch:
            tokens += _count_batch_tokens(tokenizer, batch, add_bos=add_bos, add_eos=add_eos)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    seconds = max(time.perf_counter() - start, 1e-9)
    return TokenizerBenchmarkResult(
        backend=backend,
        records=records,
        chars=chars,
        tokens=tokens,
        seconds=seconds,
        chars_per_second=chars / seconds,
        tokens_per_second=tokens / seconds,
        peak_memory_mb=peak / (1024 * 1024),
        invalid_records=stats.invalid_records,
        skipped_records=stats.skipped_records,
    )


def _count_batch_tokens(
    tokenizer: TokenizerLike,
    texts: list[str],
    *,
    add_bos: bool,
    add_eos: bool,
) -> int:
    encoded = encode_batch(tokenizer, texts, add_bos=add_bos, add_eos=add_eos)
    return sum(len(ids) for ids in encoded)
