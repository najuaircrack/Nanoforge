from __future__ import annotations

from nanoforge.imports import detect_model_format, import_model, load_registry
from nanoforge.templates import build_template_config


def test_import_registry_records_gguf(tmp_path, monkeypatch):
    registry = tmp_path / "imports.json"
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setenv("NANOFORGE_MODEL_REGISTRY", str(registry))
    entry = import_model(str(model), "tiny")
    assert entry.backend == "llama_cpp"
    assert load_registry()["tiny"].source == str(model)


def test_detect_huggingface_source():
    assert detect_model_format("mistralai/Mistral-7B-v0.1") == ("huggingface", "transformers")


def test_template_config_contains_cpu_chat_defaults():
    cfg = build_template_config("chat assistant", "8GB", "fast/small", "parquet chat")
    assert "d_model: 256" in cfg
    assert "mode: chat" in cfg
    assert "--text-column messages" in cfg


def test_auto_train_prepare_only_writes_config_and_packed_data(tmp_path, monkeypatch):
    import json

    from nanoforge.cli import main

    data = tmp_path / "raw"
    data.mkdir()
    (data / "chat.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "configs" / "chat.yaml"
    packed = tmp_path / "packed"
    main(
        [
            "auto-train",
            "--input",
            str(data),
            "--name",
            "chat",
            "--mode",
            "chat",
            "--tokenizer",
            "byte",
            "--seq-len",
            "32",
            "--packed-dir",
            str(packed),
            "--config-out",
            str(config),
            "--no-train",
        ]
    )
    assert config.exists()
    assert (packed / "train.bin").exists()
    assert (packed / "train.labels.bin").exists()
