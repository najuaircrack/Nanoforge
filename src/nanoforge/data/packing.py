from __future__ import annotations

import json
import os
import random
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from nanoforge.data.cleaning import CleaningConfig, clean_records
from nanoforge.data.formats import DatasetRecord, DatasetStats, iter_dataset_records
from nanoforge.data.tokenizer import TokenizerLike


@dataclass
class PackingStats:
    train_tokens: int = 0
    val_tokens: int = 0
    records_seen: int = 0
    records_written: int = 0
    shards: int = 0


class TokenShardWriter:
    def __init__(self, out_dir: str | Path, split: str, vocab_size: int, shard_tokens: int = 50_000_000):
        self.out_dir = Path(out_dir)
        self.split = split
        self.vocab_size = vocab_size
        self.shard_tokens = shard_tokens
        self.dtype_code = "H" if vocab_size <= 65535 else "I"
        self.dtype_name = "uint16" if self.dtype_code == "H" else "uint32"
        self.buffer = array(self.dtype_code)
        self.shard_id = 0
        self.total_tokens = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, tokens: Iterable[int]) -> None:
        for token in tokens:
            self.buffer.append(token)
            if len(self.buffer) >= self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        suffix = "" if self.shard_id == 0 else f".{self.shard_id:05d}"
        path = self.out_dir / f"{self.split}{suffix}.bin"
        with path.open("wb") as fh:
            self.buffer.tofile(fh)
        self.total_tokens += len(self.buffer)
        self.buffer = array(self.dtype_code)
        self.shard_id += 1

    def close(self) -> None:
        self.flush()
        meta = {
            "dtype": self.dtype_name,
            "tokens": self.total_tokens,
            "vocab_size": self.vocab_size,
            "shards": self.shard_id,
            "files": [f"{self.split}{'' if i == 0 else f'.{i:05d}'}.bin" for i in range(self.shard_id)],
        }
        (self.out_dir / f"{self.split}.bin.meta").write_text(
            "\n".join(f"{k}={v}" for k, v in meta.items() if k != "files") + "\n",
            encoding="utf-8",
        )
        (self.out_dir / f"{self.split}.manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def stream_tokenize_records(
    records: Iterable[DatasetRecord],
    tokenizer: TokenizerLike,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Iterator[tuple[DatasetRecord, list[int]]]:
    for record in records:
        ids = tokenizer.encode(record.text, add_bos=add_bos, add_eos=add_eos)
        if ids:
            yield record, ids


def build_packed_dataset_streaming(
    input_paths: Iterable[str | Path],
    out_dir: str | Path,
    tokenizer: TokenizerLike,
    *,
    val_fraction: float = 0.01,
    text_key: str = "text",
    code_only: bool = False,
    seed: int = 1337,
    shard_tokens: int = 50_000_000,
    cleaning: CleaningConfig | None = None,
    progress_callback: Callable[[PackingStats], None] | None = None,
) -> PackingStats:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_stats = DatasetStats()
    records = iter_dataset_records(input_paths, text_key=text_key, code_only=code_only, stats=data_stats)
    records = clean_records(records, cleaning or CleaningConfig())
    train = TokenShardWriter(out_dir, "train", tokenizer.vocab_size, shard_tokens)
    val = TokenShardWriter(out_dir, "val", tokenizer.vocab_size, shard_tokens)
    stats = PackingStats()
    first_ids: list[int] | None = None
    try:
        for record, ids in stream_tokenize_records(records, tokenizer):
            if first_ids is None:
                first_ids = ids
            stats.records_seen += 1
            if rng.random() < val_fraction:
                val.write(ids)
                stats.val_tokens += len(ids)
            else:
                train.write(ids)
                stats.train_tokens += len(ids)
            stats.records_written += 1
            if progress_callback is not None:
                progress_callback(stats)
        if first_ids is not None and stats.val_tokens == 0:
            val.write(first_ids)
            stats.val_tokens += len(first_ids)
        if first_ids is not None and stats.train_tokens == 0:
            train.write(first_ids)
            stats.train_tokens += len(first_ids)
    finally:
        train.close()
        val.close()
    stats.shards = train.shard_id + val.shard_id
    manifest = {"packing": asdict(stats), "dataset": asdict(data_stats), "pid": os.getpid()}
    (out_dir / "packing.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stats
