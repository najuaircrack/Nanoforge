from __future__ import annotations

import json
from pathlib import Path

from nanoforge.data.formats import (
    _best_turn_field,
    _infer_row_mode,
    _looks_like_turn_list,
    detect_format,
    iter_dataset_records,
)


def test_detect_format_by_suffix(tmp_path: Path) -> None:
    assert detect_format("x.jsonl") == "jsonl"
    assert detect_format("x.json") == "json"
    assert detect_format("x.csv") == "csv"
    assert detect_format("x.tsv") == "csv"
    assert detect_format("x.parquet") == "parquet"
    assert detect_format("hf://foo/bar") == "huggingface"
    assert detect_format("https://example.com/data.jsonl") == "http-jsonl"


def test_looks_like_turn_list_alternating_strings() -> None:
    turns = [
        "Can you explain what recursion means in programming?",
        "Recursion is when a function calls itself to solve a smaller subproblem.",
    ]
    assert _looks_like_turn_list(turns) is True


def test_looks_like_turn_list_rejects_short_labels() -> None:
    assert _looks_like_turn_list(["a", "b"]) is False
    assert _looks_like_turn_list(["tag1", "tag2", "tag3"]) is False


def test_looks_like_turn_list_role_dicts() -> None:
    turns = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    assert _looks_like_turn_list(turns) is True


def test_best_turn_field_picks_conversation() -> None:
    row = {
        "id": "123",
        "tags": ["short", "labels"],
        "data": [
            "A reasonably long user question about the weather forecast today.",
            "A reasonably long assistant answer describing tomorrow's sunny weather.",
        ],
    }
    assert _best_turn_field(row) == "data"


def test_infer_row_mode_value_based_chat() -> None:
    row = {"id": "1", "data": [
        "What time does the museum open on weekends normally?",
        "The museum opens at nine in the morning on both Saturday and Sunday.",
    ]}
    assert _infer_row_mode(row) == "chat"


def test_iter_records_formats_ultrachat_with_role_tags(tmp_path: Path) -> None:
    path = tmp_path / "u.jsonl"
    path.write_text(
        json.dumps({"id": "1", "data": [
            "Can you explain how photosynthesis works in plants?",
            "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
        ]}),
        encoding="utf-8",
    )
    records = list(iter_dataset_records([path]))
    assert len(records) == 1
    text = records[0].text
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert records[0].metadata["mode"] == "chat"
