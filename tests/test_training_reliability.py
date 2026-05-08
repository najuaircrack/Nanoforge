from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


def test_checkpoint_hash_and_schema_roundtrip(tmp_path):
    import torch

    from nanoforge.config import ModelConfig, NanoforgeConfig
    from nanoforge.model.transformer import NanoforgeForCausalLM
    from nanoforge.training.checkpoint import load_checkpoint, save_checkpoint, verify_checkpoint

    cfg = NanoforgeConfig(
        model=ModelConfig(vocab_size=32, max_seq_len=8, d_model=16, n_layers=1, n_heads=2)
    )
    model = NanoforgeForCausalLM(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, cfg, step=3, val_loss=1.25)
    assert verify_checkpoint(path)
    assert path.with_suffix(".pt.hash").exists()
    assert path.with_suffix(".pt.meta.json").exists()
    payload = load_checkpoint(path)
    assert payload["schema_version"] == 2
    assert payload["step"] == 3
    assert "rng" in payload
    assert "registry" in payload


def test_health_monitor_reports_entropy_and_optimizer_stats():
    import torch

    from nanoforge.training.health import TrainingHealthMonitor

    model = torch.nn.Linear(4, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 8)
    snapshot = TrainingHealthMonitor().observe(
        loss=1.0,
        grad_norm=0.5,
        logits=x,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    assert snapshot.metrics["health/logit_entropy"] > 0
    assert "health/optimizer_state_tensors" in snapshot.metrics
