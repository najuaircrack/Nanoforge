from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from nanoforge.data.cleaning import CleaningConfig
from nanoforge.data.packing import build_packed_dataset_streaming
from nanoforge.data.tokenizer import TokenizerLike, load_tokenizer


class PackedMemmapDataset:
    """Validated token shards with explicit shifted-label and sequence contracts."""

    def __init__(self, path: str | Path, seq_len: int, *, seed: int | None = None):
        import hashlib

        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.path = Path(path)
        self.seq_len = seq_len
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self.tokens: list[np.memmap] = []
        self.labels: list[np.memmap] | None = None
        manifest_path = self.path.with_name(f"{self.path.stem}.manifest.json")
        meta_path = self.path.with_suffix(self.path.suffix + ".meta")
        manifest = {}
        dtype = "uint16"
        packing_path = self.path.parent / "packing.manifest.json"
        if packing_path.exists():
            packing = json.loads(packing_path.read_text(encoding="utf-8"))
            if packing.get("status", "complete") != "complete":
                raise ValueError("Packed dataset transaction is incomplete; repack before training")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("schema_version", 1)) not in {1, 2}:
                raise ValueError("Unsupported packed dataset schema version")
            dtype = manifest.get("dtype", dtype)
        elif meta_path.exists():
            for line in meta_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("dtype="):
                    dtype = line.split("=", 1)[1].strip()
        if dtype not in {"uint16", "uint32"}:
            raise ValueError(f"Invalid token dtype: {dtype}")
        self.fixed_sequence_len = int(manifest.get("sequence_length", 0) or 0)
        self.labels_are_shifted = bool(manifest.get("labels_are_shifted", self.fixed_sequence_len > 0))
        if self.fixed_sequence_len and self.seq_len > self.fixed_sequence_len:
            raise ValueError(f"Requested seq_len={seq_len}, but packed fixed sequence length is {self.fixed_sequence_len}.")

        def files_from(description, default, allowed_dtypes):
            file_dtype = description.get("dtype", allowed_dtypes[0])
            if file_dtype not in allowed_dtypes:
                raise ValueError(f"Invalid shard dtype: {file_dtype}")
            names = description.get("files", default)
            if not names:
                raise ValueError(f"Dataset {path} contains no shards")
            result = []
            for name in names:
                file = (self.path.parent / name).resolve()
                if not file.is_relative_to(self.path.parent.resolve()):
                    raise ValueError("Shard manifest path escapes its dataset directory")
                if not file.is_file():
                    raise ValueError(f"Missing dataset shard: {name}")
                expected = description.get("file_hashes", {}).get(name)
                if expected:
                    digest = hashlib.sha256()
                    with file.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != expected:
                        raise ValueError(f"Dataset shard integrity mismatch: {name}")
                if file.stat().st_size % np.dtype(file_dtype).itemsize:
                    raise ValueError(f"Dataset shard has an invalid byte length: {name}")
                result.append(np.memmap(file, dtype=np.dtype(file_dtype), mode="r"))
            return result

        try:
            self.tokens = files_from(manifest, [self.path.name], (dtype,))
            label_path = self.path.with_name(f"{self.path.stem}.labels.manifest.json")
            if label_path.exists():
                label_manifest = json.loads(label_path.read_text(encoding="utf-8"))
                self.labels = files_from(label_manifest, [], ("int32",))
                if len(self.labels) != len(self.tokens):
                    raise ValueError("Token and label shard counts differ")
                if any(len(x) != len(y) for x, y in zip(self.tokens, self.labels)):
                    raise ValueError("Token and label shard lengths differ")
            elif manifest.get("label_files") or self.labels_are_shifted:
                raise ValueError("Missing required label shard manifest")
            self.lengths = np.asarray([len(tokens) for tokens in self.tokens], dtype=np.int64)
            if self.fixed_sequence_len:
                if any(length % self.fixed_sequence_len for length in self.lengths):
                    raise ValueError("Fixed-sequence shards do not align to sequence boundaries; repack this legacy dataset")
                valid_counts = self.lengths // self.fixed_sequence_len
            else:
                valid_counts = self.lengths - seq_len
            if any(count < 1 for count in valid_counts):
                raise ValueError(f"Dataset shard is too small for seq_len={seq_len}; repack with larger shards")
            self._sample_counts = valid_counts
            self.probs = valid_counts / valid_counts.sum()
        except Exception:
            self.close()
            raise

    def __len__(self) -> int:
        return int(self._sample_counts.sum())

    def sample(self, batch_size: int):
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rng = self._rng if self._rng is not None else np.random
        randint = self._rng.integers if self._rng is not None else np.random.randint
        shard_ids = rng.choice(len(self.tokens), size=batch_size, p=self.probs)
        starts = [
            randint(0, self._sample_counts[shard]) * self.fixed_sequence_len
            if self.fixed_sequence_len else randint(0, self._sample_counts[shard])
            for shard in shard_ids
        ]
        x = np.stack([self.tokens[shard][i:i+self.seq_len] for shard, i in zip(shard_ids, starts)])
        if self.labels is None:
            y = np.stack([self.tokens[shard][i+1:i+1+self.seq_len] for shard, i in zip(shard_ids, starts)])
        else:
            offset = 0 if self.labels_are_shifted else 1
            y = np.stack([self.labels[shard][i+offset:i+offset+self.seq_len] for shard, i in zip(shard_ids, starts)])
        if x.shape != y.shape:
            raise ValueError("Packed input and target shapes differ")
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

    def close(self) -> None:
        for value in getattr(self, "tokens", []) + (getattr(self, "labels", None) or []):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None and not mmap.closed:
                mmap.close()
        self.tokens = []
        self.labels = None

    def state_dict(self) -> dict:
        import copy

        return {"schema_version": 1, "seq_len": self.seq_len,
                "rng": copy.deepcopy(self._rng.bit_generator.state) if self._rng is not None else None}

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema_version") != 1 or state.get("seq_len") != self.seq_len:
            raise ValueError("Dataset sampler state is incompatible")
        if state.get("rng") is not None:
            if self._rng is None:
                self._rng = np.random.default_rng()
            self._rng.bit_generator.state = state["rng"]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


def build_packed_dataset(
    input_paths: Iterable[str | Path],
    out_dir: str | Path,
    tokenizer: TokenizerLike,
    val_fraction: float = 0.01,
    code_only: bool = False,
    jsonl_text_key: str = "text",
    min_chars: int = 16,
    seed: int = 1337,
    mode: str = "auto",
    loss_masking: str = "auto",
    tokenizer_batch_size: int = 256,
    progress_callback=None,
    text_columns: tuple = (),
    seq_len: int | None = None,
    governance=None,
    reader_limits=None,
    split_group_key: str = "group_id",
) -> None:
    build_packed_dataset_streaming(
        input_paths,
        out_dir,
        tokenizer,
        val_fraction=val_fraction,
        text_key=jsonl_text_key,
        code_only=code_only,
        text_columns=text_columns,
        seed=seed,
        seq_len=seq_len or 512,
        cleaning=CleaningConfig(min_chars=min_chars, deduplicate=True),
        mode=mode,
        loss_masking=loss_masking,
        tokenizer_batch_size=tokenizer_batch_size,
        progress_callback=progress_callback,
        governance=governance,
        reader_limits=reader_limits,
        split_group_key=split_group_key,
    )


def make_torch_batch(dataset: PackedMemmapDataset, batch_size: int, device: str, pin_memory: bool = False):
    x, y = dataset.sample(batch_size)
    if pin_memory:
        x = x.pin_memory()
        y = y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def prepare_from_config(config_path: str | Path, input_paths: Iterable[str | Path], out_dir: str | Path) -> None:
    from nanoforge.config import load_config

    cfg = load_config(config_path)
    tokenizer = load_tokenizer(cfg.data.tokenizer_type, cfg.data.tokenizer_path)
    build_packed_dataset(
        input_paths,
        out_dir,
        tokenizer,
        mode=cfg.data.mode,
        loss_masking="assistant_only" if cfg.data.assistant_only_loss else cfg.data.loss_masking,
        tokenizer_batch_size=cfg.data.tokenizer_batch_size,
        seq_len=cfg.data.seq_len,
        seed=cfg.training.seed,
    )
