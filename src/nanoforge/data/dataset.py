from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import numpy as np

from nanoforge.data.cleaning import CleaningConfig
from nanoforge.data.packing import build_packed_dataset_streaming
from nanoforge.data.tokenizer import TokenizerLike, load_tokenizer


class PackedMemmapDataset:
    """Random fixed-length windows over a contiguous uint16/uint32 token memmap."""

    def __init__(self, path: str | Path, seq_len: int):
        self.path = Path(path)
        self.seq_len = seq_len
        manifest_path = self.path.with_name(f"{self.path.stem}.manifest.json")
        meta_path = self.path.with_suffix(self.path.suffix + ".meta")
        dtype = "uint16"
        files = [self.path]
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dtype = manifest.get("dtype", dtype)
            files = [self.path.parent / name for name in manifest.get("files", [self.path.name])]
        elif meta_path.exists():
            for line in meta_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("dtype="):
                    dtype = line.split("=", 1)[1].strip()
        self.tokens = [np.memmap(file, dtype=np.dtype(dtype), mode="r") for file in files if file.exists()]
        self.lengths = np.asarray([len(tokens) for tokens in self.tokens], dtype=np.int64)
        valid = self.lengths > seq_len + 1
        self.tokens = [tokens for tokens, is_valid in zip(self.tokens, valid) if is_valid]
        self.lengths = self.lengths[valid]
        if not self.tokens:
            raise ValueError(f"Dataset {path} is too small for seq_len={seq_len}.")
        self.probs = self.lengths / self.lengths.sum()

    def __len__(self) -> int:
        return int(max(1, self.lengths.sum() - self.seq_len - len(self.tokens)))

    def sample(self, batch_size: int):
        import torch

        shard_ids = np.random.choice(len(self.tokens), size=(batch_size,), p=self.probs)
        starts = [
            np.random.randint(0, len(self.tokens[shard]) - self.seq_len - 1)
            for shard in shard_ids
        ]
        x = np.stack([self.tokens[shard][i : i + self.seq_len] for shard, i in zip(shard_ids, starts)])
        y = np.stack([self.tokens[shard][i + 1 : i + 1 + self.seq_len] for shard, i in zip(shard_ids, starts)])
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


def build_packed_dataset(
    input_paths: Iterable[str | Path],
    out_dir: str | Path,
    tokenizer: TokenizerLike,
    val_fraction: float = 0.01,
    code_only: bool = False,
    jsonl: bool = False,
    jsonl_text_key: str = "text",
    min_chars: int = 16,
    shuffle_docs: bool = True,
    seed: int = 1337,
    progress_callback=None,
) -> None:
    _ = jsonl, shuffle_docs
    build_packed_dataset_streaming(
        input_paths,
        out_dir,
        tokenizer,
        val_fraction=val_fraction,
        text_key=jsonl_text_key,
        code_only=code_only,
        seed=seed,
        cleaning=CleaningConfig(min_chars=min_chars, deduplicate=True),
        progress_callback=progress_callback,
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
    build_packed_dataset(input_paths, out_dir, tokenizer)
