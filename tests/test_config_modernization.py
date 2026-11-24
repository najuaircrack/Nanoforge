from __future__ import annotations

import math

import pytest

from nanoforge.config import ModelConfig, TrainConfig, load_config, save_config
from nanoforge.templates import build_cpu_config


def test_malformed_yaml_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("training:\n  max_steps: 7\nbroken: [\n")
    with pytest.raises(ValueError, match="YAML"):
        load_config(path)


def test_root_and_nested_unknown_fields_are_strict(tmp_path):
    path = tmp_path / "bad.yaml"
    for text in ("trainng:\n  max_steps: 7\n", "training:\n  misspelled: 7\n"):
        path.write_text(text)
        with pytest.raises(ValueError, match="unknown"):
            load_config(path, strict=True)


def test_compatibility_loader_warns_for_unknown_root(tmp_path, caplog):
    path = tmp_path / "legacy.yaml"
    path.write_text("trainng:\n  max_steps: 7\n")
    from nanoforge.logging_utils import get_logger
    logger = get_logger("nanoforge.config")
    logger.addHandler(caplog.handler)
    try:
        load_config(path)
    finally:
        logger.removeHandler(caplog.handler)
    assert "unknown" in caplog.text


@pytest.mark.parametrize("text", [
    "training: []", "model: []", "training:\n  max_steps: true",
    "training:\n  learning_rate: .nan", "model:\n  dropout: 2.0",
    "model:\n  n_heads: 0", "data:\n  seq_len: 0",
    "inference:\n  top_p: 1.5", "training:\n  grad_accum_steps: 0",
    "training:\n  distributed_backend: invalid",
])
def test_invalid_config_values_fail_before_runtime(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises((ValueError, TypeError)):
        load_config(path, strict=True)


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("training:\n  max_steps: 1\n  max_steps: 9\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_config(path)


def test_new_training_fields_roundtrip(tmp_path):
    from nanoforge.config import NanoforgeConfig
    training = TrainConfig(micro_batch_size=2, grad_accum_steps=3, cpu_threads=2,
                           init_from_checkpoint="checkpoints/base.pt",
                           swa_start=10, optimizer_cpu_offload=True,
                           curriculum=[{"step": 0, "seq_len": 16}])
    assert training.batch_size == 6
    path = tmp_path / "config.yaml"
    save_config(NanoforgeConfig(training=training), path)
    loaded = load_config(path, strict=True)
    assert loaded.training.curriculum == training.curriculum
    assert loaded.training.swa_start == 10


def test_failed_serialization_preserves_original(tmp_path, monkeypatch):
    import yaml

    from nanoforge.config import NanoforgeConfig
    path = tmp_path / "config.yaml"
    path.write_text("original")
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic serialization failure")
    monkeypatch.setattr(yaml, "safe_dump", fail)
    with pytest.raises(RuntimeError):
        save_config(NanoforgeConfig(), path)
    assert path.read_text() == "original"


@pytest.mark.parametrize("ram", ["4GB", "8GB", "16GB", "32GB+"])
def test_every_cpu_template_constructs_valid_gqa(tmp_path, ram):
    path = tmp_path / "config.yaml"
    path.write_text(build_cpu_config(name="synthetic", mode="chat", ram=ram,
                                    speed="fast/small", data_format="JSONL",
                                    tokenizer_type="byte", tokenizer_path="",
                                    packed_dir="packed", seq_len_override=32))
    config = load_config(path, strict=True)
    assert config.model.n_heads % config.model.n_kv_heads == 0


def test_new_attention_and_inference_fields_validate():
    from nanoforge.config import InferenceConfig
    assert ModelConfig(attention_chunk_size=64, kv_page_size=16).kv_page_size == 16
    assert InferenceConfig(sampler="contrastive", contrastive_top_k=3).sampler == "contrastive"
    with pytest.raises(ValueError):
        ModelConfig(kv_page_size=0)
    with pytest.raises(ValueError):
        TrainConfig(recovery_lr_factor=math.nan)

