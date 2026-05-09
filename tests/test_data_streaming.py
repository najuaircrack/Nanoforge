from __future__ import annotations

import json

import pytest

from nanoforge.data.formats import detect_format, inspect_dataset, iter_dataset_records
from nanoforge.data.dataset import PackedMemmapDataset
from nanoforge.data.modes import encode_training_record
from nanoforge.data.packing import build_packed_dataset_streaming
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


def test_chat_mode_masks_user_tokens_and_trains_assistant_tokens():
    from nanoforge.data.formats import DatasetRecord

    tokenizer = ByteTokenizer()
    record = DatasetRecord("<|user|>\nName?\n<|assistant|>\nNajwan.\n", "memory.jsonl", {"mode": "chat"})
    encoded = encode_training_record(record, tokenizer, mode="auto", loss_masking="auto")
    assistant_ids = tokenizer.encode("Najwan.\n", add_bos=False, add_eos=False)
    assert encoded.mode == "chat"
    assert encoded.loss_masking == "assistant_only"
    assert any(token in encoded.labels for token in assistant_ids)
    user_ids = tokenizer.encode("Name?\n", add_bos=False, add_eos=False)
    start = next(
        idx
        for idx in range(len(encoded.ids) - len(user_ids) + 1)
        if encoded.ids[idx : idx + len(user_ids)] == user_ids
    )
    assert encoded.labels[start : start + len(user_ids)] == [-100] * len(user_ids)


def test_packed_chat_dataset_writes_label_masks(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}) + "\n",
        encoding="utf-8",
    )
    stats = build_packed_dataset_streaming(
        [path],
        tmp_path / "packed",
        ByteTokenizer(),
        val_fraction=0.0,
        mode="auto",
        loss_masking="auto",
        seq_len=64,
        cleaning=None,
    )
    assert stats.records_written == 1
    assert stats.mixed_sequences >= 1
    assert stats.all_masked_sequences == 0
    assert stats.all_unmasked_sequences == 0
    assert (tmp_path / "packed" / "train.labels.manifest.json").exists()
    dataset = PackedMemmapDataset(tmp_path / "packed" / "train.bin", seq_len=64)
    _x, y = dataset.sample(4)
    unmasked = (y != -100).sum(dim=1)
    assert bool(((unmasked > 0) & (unmasked < 64)).all())


def test_parquet_messages_column_formats_stringified_chat(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "chat.parquet"
    messages = json.dumps([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])
    pq.write_table(pa.table({"messages": [messages]}), path)
    records = list(iter_dataset_records([path], text_columns=("messages",)))
    assert "<|user|>" in records[0].text
    assert "<|assistant|>" in records[0].text
    assert records[0].metadata["mode"] == "chat"
