"""Test trainer graceful shutdown on SIGINT/SIGTERM.

The trainer wraps the training loop with signal handlers that set a stop flag, check it
at the top of each step, save an interrupt checkpoint, and restore the original handlers
on exit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


def _make_tiny_dataset(tmp_path: Path, name: str, seq_len: int = 16, n_seqs: int = 8) -> Path:
    """Build a minimal fixed-sequence packed dataset (train + val) with manifests."""
    import numpy as np

    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)

    def _write(split: str, sequences: int) -> None:
        n = sequences * seq_len
        tokens = np.random.randint(0, 256, size=n, dtype=np.uint16)
        labels = tokens.astype(np.int32)
        (out / f"{split}.bin").write_bytes(tokens.tobytes())
        (out / f"{split}.labels.bin").write_bytes(labels.tobytes())
        (out / f"{split}.manifest.json").write_text(
            json.dumps({"dtype": "uint16", "files": [f"{split}.bin"], "sequence_length": seq_len})
        )
        (out / f"{split}.labels.manifest.json").write_text(
            json.dumps({"dtype": "int32", "files": [f"{split}.labels.bin"], "sequence_length": seq_len})
        )

    _write("train", n_seqs)
    _write("val", n_seqs)
    return out


def test_trainer_completes_without_interrupt(tmp_path: Path) -> None:
    """Baseline: trainer reaches max_steps and saves last.pt."""
    from nanoforge.config import DataConfig, ModelConfig, NanoforgeConfig, TrainConfig
    from nanoforge.training.trainer import Trainer

    packed = _make_tiny_dataset(tmp_path, "packed")
    cfg = NanoforgeConfig(
        model=ModelConfig(vocab_size=256, max_seq_len=16, d_model=32, n_layers=1, n_heads=2),
        training=TrainConfig(
            run_name="baseline",
            output_dir=str(tmp_path / "out"),
            max_steps=3,
            micro_batch_size=1,
            grad_accum_steps=1,
            eval_interval=10,
            save_interval=10,
            log_interval=1,
        ),
        data=DataConfig(
            train_path=str(packed / "train.bin"),
            val_path=str(packed / "val.bin"),
            seq_len=16,
        ),
    )
    trainer = Trainer(cfg)
    trainer.train()
    assert (tmp_path / "out" / "last.pt").exists()
    assert not (tmp_path / "out" / "interrupt.pt").exists()
    metrics = (tmp_path / "out" / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
    final = json.loads(metrics[-1])
    assert final["event"] == "done"
    assert final["train/final_step"] == 3


def test_trainer_saves_interrupt_checkpoint_on_stop_request(tmp_path: Path) -> None:
    """When _stop_requested is set, trainer saves interrupt.pt and breaks the loop."""
    from nanoforge.config import DataConfig, ModelConfig, NanoforgeConfig, TrainConfig
    from nanoforge.training.trainer import Trainer

    packed = _make_tiny_dataset(tmp_path, "packed")
    cfg = NanoforgeConfig(
        model=ModelConfig(vocab_size=256, max_seq_len=16, d_model=32, n_layers=1, n_heads=2),
        training=TrainConfig(
            run_name="interrupt",
            output_dir=str(tmp_path / "out"),
            max_steps=20,
            micro_batch_size=1,
            grad_accum_steps=1,
            eval_interval=100,
            save_interval=100,
            log_interval=1,
        ),
        data=DataConfig(
            train_path=str(packed / "train.bin"),
            val_path=str(packed / "val.bin"),
            seq_len=16,
        ),
    )
    trainer = Trainer(cfg)

    # Deterministically request a stop after the first step completes by hooking
    # the per-step batch fetch (called at the top of every step's inner loop).
    original_batch = trainer._batch
    seen = {"steps": 0}

    def _batch_then_stop(split, batch_size):
        if split == "train":
            seen["steps"] += 1
            if seen["steps"] >= 2:
                # Second train step reached — request stop; it triggers on the
                # NEXT loop iteration's top-of-step check.
                trainer._stop_requested = True
        return original_batch(split, batch_size)

    trainer._batch = _batch_then_stop
    trainer.train()

    assert (tmp_path / "out" / "interrupt.pt").exists()
    assert not (tmp_path / "out" / "last.pt").exists()
    metrics = (tmp_path / "out" / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
    final = json.loads(metrics[-1])
    assert final["event"] == "interrupted"
    assert final["train/final_step"] < cfg.training.max_steps
