from __future__ import annotations

import json

from nanoforge.data.formats import detect_format, inspect_dataset, iter_dataset_records
from nanoforge.data.tokenizer import ByteTokenizer
from nanoforge.data.tokenizer_metrics import evaluate_tokenizer


def test_jsonl_records_stream(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps({"text": "hello world"}) + "\n", encoding="utf-8")
    records = list(iter_dataset_records([path]))
    assert detect_format(path) == "jsonl"
    assert records[0].text.strip() == "hello world"


def test_dataset_inspection_and_tokenizer_report(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha beta gamma", encoding="utf-8")
    stats = inspect_dataset([path])
    report = evaluate_tokenizer(ByteTokenizer(), [path])
    assert stats.records == 1
    assert report.tokens > 0
    assert report.unk_rate == 0.0

