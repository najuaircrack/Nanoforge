from __future__ import annotations

import json
from pathlib import Path

from nanoforge.data.profiler import profile_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_detects_ultrachat_alternating_list(tmp_path: Path) -> None:
    path = tmp_path / "ultra.jsonl"
    _write_jsonl(
        path,
        [
            {"id": str(i), "data": [
                "What is the capital of France?",
                "The capital of France is Paris, a major European city.",
                "And its population?",
                "Paris has roughly two million residents in the city proper.",
            ]}
            for i in range(20)
        ],
    )
    profile = profile_dataset([path], limit=50)
    assert profile.detected_mode == "chat"
    assert profile.recommended_loss_masking == "assistant_only"
    assert profile.turn_field == "data"
    assert profile.role_style == "alternating-list"
    assert profile.recommended_text_columns == ("data",)
    assert profile.confidence == 1.0


def test_detects_role_dict_conversations(tmp_path: Path) -> None:
    path = tmp_path / "sharegpt.jsonl"
    _write_jsonl(
        path,
        [
            {"conversations": [
                {"from": "human", "value": "Hello there, how are you doing today?"},
                {"from": "gpt", "value": "I am doing well, thank you for asking!"},
            ]}
            for _ in range(10)
        ],
    )
    profile = profile_dataset([path], limit=50)
    assert profile.detected_mode == "chat"
    assert profile.recommended_loss_masking == "assistant_only"


def test_detects_instruct(tmp_path: Path) -> None:
    path = tmp_path / "alpaca.jsonl"
    _write_jsonl(
        path,
        [
            {"instruction": "Summarize the text.", "input": "A long passage here.", "output": "A short summary."}
            for _ in range(10)
        ],
    )
    profile = profile_dataset([path], limit=50)
    assert profile.detected_mode == "instruct"
    assert profile.recommended_loss_masking == "completion_only"


def test_detects_plain_generative(tmp_path: Path) -> None:
    path = tmp_path / "plain.jsonl"
    _write_jsonl(
        path,
        [{"text": "Just a paragraph of ordinary prose that has no chat structure at all."} for _ in range(10)],
    )
    profile = profile_dataset([path], limit=50)
    assert profile.detected_mode == "generative"
    assert profile.recommended_loss_masking == "none"


def test_empty_dataset_defaults_generative(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    profile = profile_dataset([path], limit=50)
    assert profile.detected_mode == "generative"
    assert profile.num_sampled == 0


def test_to_dict_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "ultra.jsonl"
    _write_jsonl(path, [{"id": "1", "data": ["Hi there friend", "Hello, how can I help you today?"]}])
    profile = profile_dataset([path], limit=10)
    d = profile.to_dict()
    assert d["detected_mode"] == profile.detected_mode
    assert d["recommended_text_columns"] == list(profile.recommended_text_columns)
