from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


def test_tiny_forward() -> None:
    import torch

    from nanoforge.config import ModelConfig
    from nanoforge.model.transformer import NanoforgeForCausalLM

    cfg = ModelConfig(vocab_size=260, max_seq_len=32, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = NanoforgeForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, labels=x)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is not None


def test_configure_named_optimizer() -> None:
    import torch

    from nanoforge.training.utils import configure_named_optimizer

    model = torch.nn.Sequential(torch.nn.Embedding(16, 8), torch.nn.Linear(8, 16))
    opt = configure_named_optimizer(model, "adamw", lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    assert isinstance(opt, torch.optim.AdamW)
