from __future__ import annotations

import json
import os
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from nanoforge.data.cleaning import CleaningConfig, clean_records
from nanoforge.data.fim import validate_fim_tokenizer
from nanoforge.data.formats import DatasetRecord, DatasetStats, iter_dataset_records
from nanoforge.data.modes import (
    encode_training_record,
    encode_training_sequences,
    infer_record_mode,
    resolve_loss_masking,
)
from nanoforge.data.native_tokenizer import encode_batch
from nanoforge.data.tokenizer import TokenizerLike


@dataclass
class PackingStats:
    train_tokens: int = 0
    val_tokens: int = 0
    records_seen: int = 0
    records_written: int = 0
    records_skipped: int = 0
    sequences_written: int = 0
    mixed_sequences: int = 0
    all_masked_sequences: int = 0
    all_unmasked_sequences: int = 0
    masked_labels: int = 0
    unmasked_labels: int = 0
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
        metadata: dict[str, object] | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.split = split
        self.vocab_size = vocab_size
        self.shard_tokens = shard_tokens
        self.dtype_code = dtype_code or ("H" if vocab_size <= 65535 else "I")
        self.dtype_name = dtype_name or ("uint16" if self.dtype_code == "H" else "uint32")
        self.metadata = metadata or {}
        self.buffer = array(self.dtype_code)
        self.shard_id = 0
        self.total_tokens = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, tokens: Iterable[int]) -> None:
        for token in tokens:
            if (self.dtype_name == "int32" and token != -100 and not 0 <= token < self.vocab_size) or (self.dtype_name != "int32" and not 0 <= token < self.vocab_size):
                raise ValueError("Token or label ID is outside the tokenizer vocabulary")
            self.buffer.append(token)
            if len(self.buffer) >= self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        suffix = "" if self.shard_id == 0 else f".{self.shard_id:05d}"
        path = self.out_dir / f"{self.split}{suffix}.bin"
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as fh:
            self.buffer.tofile(fh)
        os.replace(temporary, path)
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
            **self.metadata,
        }
        import hashlib
        meta["file_hashes"] = {}
        for name in meta["files"]:
            digest = hashlib.sha256()
            with (self.out_dir / name).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            meta["file_hashes"][name] = digest.hexdigest()
        (self.out_dir / f"{self.split}.bin.meta").write_text(
            "\n".join(f"{k}={v}" for k, v in meta.items() if k != "files") + "\n",
            encoding="utf-8",
        )
        _atomic_json(self.out_dir / f"{self.split}.manifest.json", meta)


def stream_tokenize_records(
    records: Iterable[DatasetRecord],
    tokenizer: TokenizerLike,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
    mode: str = "auto",
    loss_masking: str = "none",
    batch_size: int = 256,
    batch_bytes: int = 4_194_304,
) -> Iterator[tuple[DatasetRecord, list[int], list[int]]]:
    if batch_size < 1 or batch_bytes < 1:
        raise ValueError("Tokenizer batching limits must be positive")
    pending_records: list[DatasetRecord] = []
    pending_texts: list[str] = []
    pending_bytes = 0

    def flush_pending() -> Iterator[tuple[DatasetRecord, list[int], list[int]]]:
        nonlocal pending_records, pending_texts, pending_bytes
        if not pending_records:
            return
        yield from _encode_plain_batch(pending_records, pending_texts, tokenizer, add_bos, add_eos)
        pending_records, pending_texts = [], []
        pending_bytes = 0

    for record in records:
        actual_mode = infer_record_mode(record, mode)
        policy = resolve_loss_masking(actual_mode, loss_masking)
        can_batch = policy == "none" and actual_mode in {"generative", "code", "creative"} and batch_size > 1
        if can_batch:
            size = len(record.text.encode("utf-8"))
            if pending_records and pending_bytes + size > batch_bytes:
                yield from flush_pending()
            pending_records.append(record)
            pending_texts.append(record.text)
            pending_bytes += size
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
        if not ids:
            continue
        labels = list(ids)
        labels[0] = -100
        yield record, ids, labels


def _observe_sequence_labels(stats: PackingStats, labels: list[int]) -> None:
    masked = sum(1 for label in labels if label == -100)
    unmasked = len(labels) - masked
    stats.masked_labels += masked
    stats.unmasked_labels += unmasked
    if masked == len(labels):
        stats.all_masked_sequences += 1
    elif unmasked == len(labels):
        stats.all_unmasked_sequences += 1
    else:
        stats.mixed_sequences += 1


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def tokenizer_identity(tokenizer: TokenizerLike) -> dict[str, object]:
    import hashlib

    payload: dict[str, object] = {
        "type": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size,
        "bos_id": tokenizer.bos_id, "eos_id": tokenizer.eos_id,
        "pad_id": tokenizer.pad_id, "unk_id": tokenizer.unk_id,
    }
    artifact_version = getattr(tokenizer, "artifact_version", None)
    if artifact_version is not None:
        payload["artifact_version"] = artifact_version
    path = getattr(tokenizer, "path", None)
    if path is not None and Path(path).is_file():
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        payload["artifact_fingerprint"] = digest.hexdigest()
    if hasattr(tokenizer, "merges"):
        payload["merges"] = [list(pair) for pair in tokenizer.merges]
        payload["special_tokens"] = list(tokenizer.special_tokens)
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def build_packed_dataset_streaming(
    input_paths: Iterable[str | Path], out_dir: str | Path, tokenizer: TokenizerLike, *,
    val_fraction: float = 0.01, text_key: str = "text", text_columns: tuple = (),
    code_only: bool = False, seed: int = 1337, seq_len: int = 512,
    shard_tokens: int = 50_000_000, cleaning: CleaningConfig | None = None,
    mode: str = "auto", loss_masking: str = "auto", tokenizer_batch_size: int = 256,
    progress_callback: Callable[[PackingStats], None] | None = None,
    governance=None, reader_limits=None, split_group_key: str = "group_id",
    records: Iterable[DatasetRecord] | None = None,
) -> PackingStats:
    import hashlib
    import sqlite3
    import tempfile

    from nanoforge.data.cleaning import CleaningStats
    from nanoforge.data.formats import ReaderLimits
    from nanoforge.data.governance import DataGovernancePolicy, GovernanceStats, govern_records

    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if seq_len < 2 or shard_tokens < 1 or tokenizer_batch_size < 1:
        raise ValueError("Sequence length, shard size and tokenizer batch size must be positive")
    normalized_mode = mode.strip().lower().replace("-", "_")
    if normalized_mode == "fim":
        validate_fim_tokenizer(tokenizer)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir / "packing.manifest.json", {"schema_version": 2, "status": "preparing"})
    data_stats = DatasetStats()
    cleaning_config = cleaning or CleaningConfig()
    cleaning_stats = CleaningStats()
    governance_policy = governance or DataGovernancePolicy()
    governance_stats = GovernanceStats()
    custom_records = records is not None
    raw_records = records if custom_records else iter_dataset_records(
        input_paths, text_key=text_key, text_columns=text_columns, code_only=code_only,
        stats=data_stats, limits=reader_limits,
        group_key=split_group_key,
    )

    def observe_custom():
        for record in raw_records:
            if custom_records:
                data_stats.records += 1
                data_stats.bytes_read += len(record.text.encode("utf-8"))
            yield record

    prepared = clean_records(
        govern_records(observe_custom(), governance_policy, governance_stats),
        cleaning_config, stats=cleaning_stats,
    )
    normalized_masking = loss_masking.strip().lower().replace("-", "_")
    boundary_packing = normalized_mode in {"auto", "chat", "instruct", "reasoning", "roleplay", "fim"} or normalized_masking in {
        "assistant_only", "completion_only",
    }
    aligned_shard_tokens = max(seq_len, shard_tokens // seq_len * seq_len) if boundary_packing else max(seq_len + 1, shard_tokens)
    metadata = {
        "schema_version": 2, "fixed_sequences": boundary_packing,
        "sequence_length": seq_len if boundary_packing else 0,
        "labels_are_shifted": boundary_packing, "label_files": True,
    }
    writers = {
        split: TokenShardWriter(out_dir, split, tokenizer.vocab_size, aligned_shard_tokens, metadata=metadata)
        for split in ("train", "val")
    }
    labels = {
        split: TokenShardWriter(out_dir, split + ".labels", tokenizer.vocab_size, aligned_shard_tokens,
                                dtype_code="i", dtype_name="int32", metadata=metadata)
        for split in ("train", "val")
    }
    stats = PackingStats()
    group_samples = {"train": [], "val": []}
    group_counts = {"train": 0, "val": 0}
    group_count = 0
    split_manifest = {
        "strategy": "content-group-seeded-with-disjoint-reserved-first-two-v1",
        "seed": seed, "val_fraction": val_fraction, "group_key": split_group_key,
        "overlap_groups": 0, "train_groups": group_samples["train"],
        "val_groups": group_samples["val"], "groups_file": "split.groups.jsonl",
    }
    groups_tmp = out_dir / "split.groups.jsonl.tmp"
    try:
        with tempfile.TemporaryDirectory(prefix="nanoforge-splits-") as temp, groups_tmp.open("w", encoding="utf-8") as group_file:
            connection = sqlite3.connect(str(Path(temp) / "groups.sqlite"))
            connection.execute("create table groups (id text primary key, split text not null)")
            connection.execute("pragma cache_size=-2048")

            def split_for(record):
                nonlocal group_count
                raw_group = record.metadata.get(split_group_key) or hashlib.sha256(record.text.encode("utf-8")).hexdigest()
                group_id = hashlib.sha256(str(raw_group).encode("utf-8")).hexdigest()
                existing = connection.execute("select split from groups where id=?", (group_id,)).fetchone()
                if existing:
                    return existing[0]
                if val_fraction == 0:
                    split = "train"
                elif group_count == 0:
                    split = "val"
                elif group_count == 1:
                    split = "train"
                else:
                    random_value = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest(), 16) / (1 << 256)
                    split = "val" if random_value < val_fraction else "train"
                group_count += 1
                group_counts[split] += 1
                connection.execute("insert into groups values (?,?)", (group_id, split))
                group_file.write(json.dumps({"group_id": group_id, "split": split}) + "\n")
                if len(group_samples[split]) < 4096:
                    group_samples[split].append(group_id)
                return split

            def write_record(record, sequences):
                stats.records_seen += 1
                valid = [(ids, target) for ids, target in sequences if ids and any(label != -100 for label in target)]
                stats.all_masked_sequences += len(sequences) - len(valid)
                if not valid:
                    stats.records_skipped += 1
                    return
                split = split_for(record)
                stats.records_written += 1
                for ids, target in valid:
                    _observe_sequence_labels(stats, target)
                    writers[split].write(ids)
                    labels[split].write(target)
                    if split == "val":
                        stats.val_tokens += len(ids)
                    else:
                        stats.train_tokens += len(ids)
                    stats.sequences_written += 1
                if progress_callback is not None:
                    progress_callback(stats)

            try:
                if boundary_packing:
                    for record in prepared:
                        encoded = encode_training_sequences(record, tokenizer, seq_len=seq_len, mode=mode, loss_masking=loss_masking)
                        write_record(record, [(sequence.input_ids, sequence.labels) for sequence in encoded])
                else:
                    for record, ids, target in stream_tokenize_records(prepared, tokenizer, mode=mode,
                                                                     loss_masking=loss_masking, batch_size=tokenizer_batch_size):
                        write_record(record, [(ids, target)])
                if stats.train_tokens == 0:
                    raise ValueError("No independent training split; provide at least two groups or set val_fraction=0")
                if val_fraction > 0 and stats.val_tokens == 0:
                    raise ValueError("No held-out validation groups; provide another independent group")
                connection.commit()
            finally:
                connection.close()
        for writer in list(writers.values()) + list(labels.values()):
            writer.close()
        os.replace(groups_tmp, out_dir / "split.groups.jsonl")
        split_manifest["group_counts"] = group_counts
        split_manifest["group_lists_truncated"] = any(count > 4096 for count in group_counts.values())
        stats.shards = sum(writer.shard_id for writer in writers.values())
        # Source paths are portable labels; content hashes carry identity.
        sources = [
            {"source": Path(source).name, "fingerprint": fingerprint}
            for source, fingerprint in data_stats.fingerprints.items()
        ]
        dataset_summary = asdict(data_stats)
        for name in ("schemas", "text_columns"):
            dataset_summary[name] = {Path(source).name: value for source, value in dataset_summary[name].items()}
        dataset_summary["fingerprints"] = {entry["source"]: entry["fingerprint"] for entry in sources}
        dataset_summary["issues"] = [{"kind": issue.kind, "message": issue.message,
                                       "source": Path(issue.source).name} for issue in data_stats.issues]
        _atomic_json(out_dir / "packing.manifest.json", {
            "schema_version": 2, "status": "complete", "packing": asdict(stats),
            "dataset": dataset_summary, "mode": mode, "loss_masking": loss_masking,
            "label_files": True, "fixed_sequences": boundary_packing,
            "sequence_length": seq_len if boundary_packing else 0,
            "tokenizer_batch_size": tokenizer_batch_size, "tokenizer": tokenizer_identity(tokenizer),
            "split": split_manifest, "cleaning": asdict(cleaning_config),
            "filtering": asdict(cleaning_stats), "governance": asdict(governance_policy),
            "governance_counts": asdict(governance_stats), "sources": sources,
            "reader": asdict(reader_limits or ReaderLimits()),
        })
        return stats
    except Exception:
        _atomic_json(out_dir / "packing.manifest.json", {
            "schema_version": 2, "status": "failed", "packing": asdict(stats),
            "failure": "Packing did not complete; partial shards must not be consumed",
        })
        raise
