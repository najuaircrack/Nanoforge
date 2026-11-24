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
        for source_index, src in enumerate(sources):
            if not np.isfinite(src.weight) or src.weight < 0:
                raise ValueError("Mixture weights must be finite and nonnegative")
            if src.weight <= 0:
                continue
            if src.seq_len is not None and src.seq_len != seq_len:
                raise ValueError("Mixture source sequence lengths must match the batch sequence length")
            ds = PackedMemmapDataset(src.path, seq_len, seed=seed + source_index + 1)
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
        perm = torch.from_numpy(self._rng.permutation(x.shape[0]))
        return x[perm], y[perm]

    def state_dict(self) -> dict:
        import copy

        return {"schema_version": 1, "names": list(self.names), "seq_len": self.seq_len,
                "rng": copy.deepcopy(self._rng.bit_generator.state),
                "datasets": [dataset.state_dict() for dataset in self.datasets]}

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema_version") != 1 or state.get("names") != self.names or state.get("seq_len") != self.seq_len:
            raise ValueError("Mixture sampler state is incompatible")
        if len(state.get("datasets", [])) != len(self.datasets):
            raise ValueError("Mixture sampler dataset count differs")
        self._rng.bit_generator.state = state["rng"]
        for dataset, dataset_state in zip(self.datasets, state["datasets"]):
            dataset.load_state_dict(dataset_state)

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()

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
