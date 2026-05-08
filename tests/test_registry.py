from __future__ import annotations

import pytest

from nanoforge.config import DataConfig, ModelConfig, TrainConfig
from nanoforge.registry import ATTENTION_BACKENDS, registry_snapshot


def test_registry_snapshot_contains_core_components():
    snapshot = registry_snapshot()
    assert "sdpa" in snapshot["attention"]
    assert "swiglu" in snapshot["ffn"]
    assert "rmsnorm" in snapshot["normalization"]
    assert "byte" in snapshot["tokenizer"]
    assert "adamw" in snapshot["optimizer"]


def test_registry_rejects_unknown_model_key():
    with pytest.raises(KeyError):
        ModelConfig(attention_backend="missing_attention")


def test_config_accepts_registry_alias_style_keys():
    cfg = DataConfig(tokenizer_type="byte-native")
    assert cfg.tokenizer_type == "byte-native"
    train = TrainConfig(optimizer="sophiag")
    assert train.optimizer == "sophiag"


def test_registered_attention_factory_is_lazy_resolvable():
    factory = ATTENTION_BACKENDS.get("sdpa")
    assert factory.__name__ == "CausalSelfAttention"
