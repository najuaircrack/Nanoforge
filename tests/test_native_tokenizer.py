from __future__ import annotations

import json

from nanoforge.data.native_tokenizer import NativeByteTokenizer, native_tokenizer_status
from nanoforge.data.tokenizer import ByteTokenizer, load_tokenizer, train_python_bpe_tokenizer
from nanoforge.data.tokenizer_benchmark import benchmark_tokenizer


def test_native_byte_tokenizer_fallback_matches_byte_tokenizer():
    text = "hello \ufffd world"
    native = NativeByteTokenizer()
    byte = ByteTokenizer()
    assert native.encode(text, add_bos=True, add_eos=True) == byte.encode(
        text,
        add_bos=True,
        add_eos=True,
    )
    assert native.decode(byte.encode(text)) == text


def test_load_byte_native_tokenizer_without_extension_is_usable():
    tokenizer = load_tokenizer("byte-native")
    ids = tokenizer.encode("abc")
    assert tokenizer.decode(ids) == "abc"


def test_native_tokenizer_status_shape():
    status = native_tokenizer_status()
    assert isinstance(status.available, bool)
    assert status.backend


def test_tokenizer_benchmark_reports_throughput(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps({"text": "alpha beta gamma"}) + "\n", encoding="utf-8")
    result = benchmark_tokenizer(NativeByteTokenizer(), [path], limit=10, batch_size=2)
    assert result.records == 1
    assert result.tokens > 0
    assert result.chars_per_second > 0


def test_python_bpe_trains_and_roundtrips_without_tokenizers_dependency(tmp_path):
    path = tmp_path / "sample.jsonl"
    tok_path = tmp_path / "python-bpe.json"
    text = "alpha alpha beta <|assistant|>"
    path.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    report = train_python_bpe_tokenizer([path], tok_path, vocab_size=280, min_frequency=2)
    tokenizer = load_tokenizer("python-bpe", tok_path)
    assert report.records == 1
    assert tok_path.exists()
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_native_bpe_loader_uses_python_fallback_when_extension_is_missing(tmp_path):
    path = tmp_path / "sample.jsonl"
    tok_path = tmp_path / "native-bpe.json"
    text = "chat assistant response"
    path.write_text(json.dumps({"text": text}) + "\n", encoding="utf-8")
    train_python_bpe_tokenizer([path], tok_path, vocab_size=272, min_frequency=2)
    tokenizer = load_tokenizer("native-bpe", tok_path)
    assert tokenizer.decode(tokenizer.encode(text)) == text
