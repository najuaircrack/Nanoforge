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
from nanoforge.data.modes import encode_training_record, infer_record_mode, resolve_loss_masking
from nanoforge.data.native_tokenizer import encode_batch
from nanoforge.data.tokenizer import TokenizerLike


@dataclass
class PackingStats:
    train_tokens: int = 0
    val_tokens: int = 0
    records_seen: int = 0
    records_written: int = 0
    shards: int = 0


class TokenShardWriter:
    def __init__(
        self,
        out_dir: str | Path,
        split: str,
        vocab_size: int,
        shard_tokens: int = 50_000_000,
        *,
        dtype_code: str | None = None,
        dtype_name: str | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.split = split
        self.vocab_size = vocab_size
        self.shard_tokens = shard_tokens
        self.dtype_code = dtype_code or ("H" if vocab_size <= 65535 else "I")
        self.dtype_name = dtype_name or ("uint16" if self.dtype_code == "H" else "uint32")
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
    mode: str = "auto",
    loss_masking: str = "none",
    batch_size: int = 256,
) -> Iterator[tuple[DatasetRecord, list[int], list[int]]]:
    pending_records: list[DatasetRecord] = []
    pending_texts: list[str] = []

    def flush_pending() -> Iterator[tuple[DatasetRecord, list[int], list[int]]]:
        nonlocal pending_records, pending_texts
        if not pending_records:
            return
        yield from _encode_plain_batch(pending_records, pending_texts, tokenizer, add_bos, add_eos)
        pending_records, pending_texts = [], []

    for record in records:
        actual_mode = infer_record_mode(record, mode)
        policy = resolve_loss_masking(actual_mode, loss_masking)
        can_batch = policy == "none" and actual_mode in {"generative", "code", "creative"} and batch_size > 1
        if can_batch:
            pending_records.append(record)
            pending_texts.append(record.text)
            if len(pending_records) >= batch_size:
                yield from flush_pending()
            continue
        yield from flush_pending()
        encoded = encode_training_record(
            record,
            tokenizer,
            mode=actual_mode,
            loss_masking=policy,
            add_bos=add_bos,
            add_eos=add_eos,
        )
        if encoded.ids:
            yield record, encoded.ids, encoded.labels
    yield from flush_pending()


def _encode_plain_batch(
    records: list[DatasetRecord],
    texts: list[str],
    tokenizer: TokenizerLike,
    add_bos: bool,
    add_eos: bool,
) -> Iterator[tuple[DatasetRecord, list[int], list[int]]]:
    for record, ids in zip(records, encode_batch(tokenizer, texts, add_bos=add_bos, add_eos=add_eos)):
        if ids:
            labels = list(ids)
            labels[0] = -100
            yield record, ids, labels


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
    mode: str = "auto",
    loss_masking: str = "auto",
    tokenizer_batch_size: int = 256,
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
    write_labels = loss_masking != "none" or mode not in {"generative", "code", "creative"}
    train_labels = (
        TokenShardWriter(out_dir, "train.labels", tokenizer.vocab_size, shard_tokens, dtype_code="i", dtype_name="int32")
        if write_labels
        else None
    )
    val_labels = (
        TokenShardWriter(out_dir, "val.labels", tokenizer.vocab_size, shard_tokens, dtype_code="i", dtype_name="int32")
        if write_labels
        else None
    )
    stats = PackingStats()
    first_ids: list[int] | None = None
    first_labels: list[int] | None = None
    try:
        for record, ids, labels in stream_tokenize_records(
            records,
            tokenizer,
            mode=mode,
            loss_masking=loss_masking,
            batch_size=tokenizer_batch_size,
        ):
            if first_ids is None:
                first_ids = ids
                first_labels = labels
            stats.records_seen += 1
            if rng.random() < val_fraction:
                val.write(ids)
                if val_labels is not None:
                    val_labels.write(labels)
                stats.val_tokens += len(ids)
            else:
                train.write(ids)
                if train_labels is not None:
                    train_labels.write(labels)
                stats.train_tokens += len(ids)
            stats.records_written += 1
            if progress_callback is not None:
                progress_callback(stats)
        if first_ids is not None and stats.val_tokens == 0:
            val.write(first_ids)
            if val_labels is not None and first_labels is not None:
                val_labels.write(first_labels)
            stats.val_tokens += len(first_ids)
        if first_ids is not None and stats.train_tokens == 0:
            train.write(first_ids)
            if train_labels is not None and first_labels is not None:
                train_labels.write(first_labels)
            stats.train_tokens += len(first_ids)
    finally:
        train.close()
        val.close()
        if train_labels is not None:
            train_labels.close()
        if val_labels is not None:
            val_labels.close()
    stats.shards = train.shard_id + val.shard_id
    manifest = {
        "packing": asdict(stats),
        "dataset": asdict(data_stats),
        "mode": mode,
        "loss_masking": loss_masking,
        "label_files": write_labels,
        "tokenizer_batch_size": tokenizer_batch_size,
        "pid": os.getpid(),
    }
    (out_dir / "packing.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return stats
