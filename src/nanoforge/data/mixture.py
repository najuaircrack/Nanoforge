from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nanoforge.data.dataset import PackedMemmapDataset


@dataclass
class MixtureSource:
    """One weighted component of a training data mixture."""

    path: str
    weight: float = 1.0
    seq_len: int | None = None
    name: str | None = None


class DataMixture:
    """Samples batches from several packed datasets according to fixed weights.

    Exposes the same ``sample(batch_size)`` surface as :class:`PackedMemmapDataset`,
    so it is a drop-in replacement inside the trainer. Each row in a batch independently
    draws its source dataset from the normalized weight distribution, which keeps the
    long-run token ratio equal to the configured weights regardless of shard sizes.
    """

    def __init__(self, sources: list[MixtureSource], seq_len: int, seed: int = 1337):
        if not sources:
            raise ValueError("DataMixture requires at least one source.")
        self.seq_len = seq_len
        self._rng = np.random.default_rng(seed)
        self.datasets: list[PackedMemmapDataset] = []
        weights: list[float] = []
        self.names: list[str] = []
        for src in sources:
            if src.weight <= 0:
                continue
            ds = PackedMemmapDataset(src.path, src.seq_len or seq_len)
            self.datasets.append(ds)
            weights.append(float(src.weight))
            self.names.append(src.name or Path(src.path).stem)
        if not self.datasets:
            raise ValueError("DataMixture has no sources with positive weight.")
        total = sum(weights)
        self.probs = np.asarray([w / total for w in weights], dtype=np.float64)

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.datasets)

    def sample(self, batch_size: int):
        import torch

        # Draw how many rows each source contributes, preserving the exact batch size.
        counts = self._rng.multinomial(batch_size, self.probs)
        xs, ys = [], []
        for ds_idx, count in enumerate(counts):
            if count == 0:
                continue
            x, y = self.datasets[ds_idx].sample(int(count))
            xs.append(x)
            ys.append(y)
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        # Shuffle so sources are interleaved within the batch, not block-ordered.
        perm = torch.randperm(x.shape[0])
        return x[perm], y[perm]

    def describe(self) -> dict[str, float]:
        return {name: float(prob) for name, prob in zip(self.names, self.probs)}


def build_mixture_from_config(data_cfg, seq_len: int, seed: int = 1337) -> DataMixture | None:
    """Build a :class:`DataMixture` from ``DataConfig.sources`` if present, else None."""

    sources = getattr(data_cfg, "sources", None)
    if not sources:
        return None
    parsed: list[MixtureSource] = []
    for entry in sources:
        if isinstance(entry, MixtureSource):
            parsed.append(entry)
        elif isinstance(entry, dict):
            parsed.append(
                MixtureSource(
                    path=entry["path"],
                    weight=float(entry.get("weight", 1.0)),
                    seq_len=entry.get("seq_len"),
                    name=entry.get("name"),
                )
            )
        else:
            raise TypeError(f"Unsupported mixture source entry: {entry!r}")
    return DataMixture(parsed, seq_len, seed=seed)
