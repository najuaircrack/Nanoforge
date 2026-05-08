from __future__ import annotations

import json

import pytest

from nanoforge.data.formats import detect_format, inspect_dataset, iter_dataset_records
from nanoforge.data.tokenizer import ByteTokenizer
from nanoforge.data.tokenizer import iter_tokenizer_training_texts, train_bpe_tokenizer
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


def test_malformed_utf8_is_recovered(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"hello\xffworld")
    stats = inspect_dataset([path])
    records = list(iter_dataset_records([path]))
    assert records[0].text.strip() == "hello\ufffdworld"
    assert stats.invalid_records == 1


def test_sharegpt_conversation_records_stream(tmp_path):
    path = tmp_path / "sharegpt.jsonl"
    row = {
        "conversations": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi"},
        ]
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    records = list(iter_dataset_records([path]))
    assert "<|user|>" in records[0].text
    assert "<|assistant|>" in records[0].text


def test_tokenizer_training_iterator_uses_structured_reader(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"prompt": "A", "completion": "B"}) + "\n", encoding="utf-8")
    texts = list(iter_tokenizer_training_texts([path], text_key="missing"))
    assert texts == ["A\nB\n"]


def test_bpe_tokenizer_dry_run_reports_malformed_records(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"abc\xffdef")
    report = train_bpe_tokenizer([path], tmp_path / "tok.json", vocab_size=32, dry_run=True)
    assert report.records == 1
    assert report.invalid_records == 1
    assert report.chars > 0


def test_parquet_reader_extracts_text_column_without_binary_tokenization(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"text": ["alpha"], "label": [1]}), path)
    stats = inspect_dataset([path])
    records = list(iter_dataset_records([path]))
    assert detect_format(path) == "parquet"
    assert records[0].text.strip() == "alpha"
    assert stats.text_columns[str(path)] == ["text"]
