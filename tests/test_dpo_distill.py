"""End-to-end tests for DPO and knowledge-distillation trainers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


def _make_tiny_dataset(tmp_path: Path, name: str, seq_len: int = 16, n_seqs: int = 8) -> Path:
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


def test_dpo_trainer_runs_and_saves(tmp_path: Path) -> None:
    from nanoforge.config import DataConfig, ModelConfig, NanoforgeConfig, TrainConfig
    from nanoforge.training.dpo_trainer import DPOTrainer

    # Preference data
    pref = tmp_path / "prefs.jsonl"
    pref.write_text(
        "\n".join(
            json.dumps({"chosen": f"good answer {i}", "rejected": f"bad {i}"})
            for i in range(8)
        ),
        encoding="utf-8",
    )

    cfg = NanoforgeConfig(
        model=ModelConfig(vocab_size=256, max_seq_len=32, d_model=32, n_layers=1, n_heads=2),
        training=TrainConfig(
            run_name="dpo",
            output_dir=str(tmp_path / "out"),
            max_steps=3,
            micro_batch_size=2,
            grad_accum_steps=1,
            log_interval=1,
            save_interval=100,
        ),
        data=DataConfig(train_path="x", val_path="x", seq_len=32, tokenizer_type="byte"),
    )
    DPOTrainer(cfg, pref, beta=0.1).train()
    assert (tmp_path / "out" / "last.pt").exists()
    metrics = (tmp_path / "out" / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
    # Should have logged a dpo/loss at some point
    assert any("dpo/loss" in line for line in metrics)
    final = json.loads(metrics[-1])
    assert final["event"] == "done"


def test_dpo_reference_is_frozen(tmp_path: Path) -> None:
    from nanoforge.config import DataConfig, ModelConfig, NanoforgeConfig, TrainConfig
    from nanoforge.training.dpo_trainer import DPOTrainer

    pref = tmp_path / "prefs.jsonl"
    pref.write_text(json.dumps({"chosen": "yes", "rejected": "no"}) + "\n", encoding="utf-8")
    cfg = NanoforgeConfig(
        model=ModelConfig(vocab_size=256, max_seq_len=32, d_model=32, n_layers=1, n_heads=2),
        training=TrainConfig(run_name="dpo", output_dir=str(tmp_path / "out"), max_steps=1, micro_batch_size=1),
        data=DataConfig(train_path="x", val_path="x", seq_len=32, tokenizer_type="byte"),
    )
    trainer = DPOTrainer(cfg, pref, beta=0.1)
    assert all(not p.requires_grad for p in trainer.ref_model.parameters())


def test_distillation_trainer_runs(tmp_path: Path) -> None:
    from nanoforge.config import DataConfig, ModelConfig, NanoforgeConfig, TrainConfig
    from nanoforge.training.distillation_trainer import DistillationTrainer

    packed = _make_tiny_dataset(tmp_path, "packed")
    cfg = NanoforgeConfig(
        model=ModelConfig(vocab_size=256, max_seq_len=16, d_model=32, n_layers=1, n_heads=2),
        training=TrainConfig(
            run_name="distill",
            output_dir=str(tmp_path / "out"),
            max_steps=3,
            micro_batch_size=1,
            grad_accum_steps=1,
            eval_interval=100,
            save_interval=100,
            log_interval=1,
            distill_alpha=0.5,
            distill_temperature=2.0,
        ),
        data=DataConfig(train_path=str(packed / "train.bin"), val_path=str(packed / "val.bin"), seq_len=16),
    )
    trainer = DistillationTrainer(cfg)
    # Teacher must be frozen
    assert all(not p.requires_grad for p in trainer.teacher.parameters())
    trainer.train()
    assert (tmp_path / "out" / "last.pt").exists()
    metrics = (tmp_path / "out" / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
    final = json.loads(metrics[-1])
    assert final["event"] == "done"
