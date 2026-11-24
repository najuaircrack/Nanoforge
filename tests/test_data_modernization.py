from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from nanoforge.data.cleaning import CleaningConfig
from nanoforge.data.dataset import PackedMemmapDataset
from nanoforge.data.fim import transform_fim_text
from nanoforge.data.formats import (
    DatasetRecord,
    DatasetStats,
    dataset_fingerprint,
    iter_dataset_records,
)
from nanoforge.data.modes import (
    _fixed_shifted_sequences,
    encode_training_record,
    encode_training_sequences,
)
from nanoforge.data.native_tokenizer import encode_batch
from nanoforge.data.packing import build_packed_dataset_streaming, tokenizer_identity
from nanoforge.data.tokenizer import ByteTokenizer, PurePythonBPETokenizer, load_tokenizer


def _source(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _pack(tmp_path, rows, **kwargs):
    path = _source(tmp_path, rows)
    out = tmp_path / "packed"
    stats = build_packed_dataset_streaming(
        [path], out, ByteTokenizer(), cleaning=CleaningConfig(min_chars=1),
        seq_len=16, val_fraction=0, **kwargs,
    )
    return out, stats


@pytest.mark.parametrize("tokenizer", [ByteTokenizer(), PurePythonBPETokenizer()])
def test_generic_batch_supports_python_tokenizers(tokenizer):
    texts = ["synthetic example", "second example"]
    assert encode_batch(tokenizer, texts, add_bos=True) == [
        tokenizer.encode(text, add_bos=True) for text in texts
    ]


def test_nonboundary_labels_are_next_token_targets(tmp_path):
    out, _ = _pack(tmp_path, [{"text": "abcdefghijklmnopqrstuvwxyz" * 8}],
                   mode="generative", loss_masking="auto")
    with PackedMemmapDataset(out / "train.bin", 16) as dataset:
        np.random.seed(2)
        x, y = dataset.sample(4)
        assert bool((x[:, 1:] == y[:, :-1]).all())
        assert not bool((x == y).all())


def test_fixed_shards_preserve_sequences_and_all_targets(tmp_path):
    out, stats = _pack(tmp_path, [{"text": "abcdefghijklmnopqrstuvwxyz" * 4}], shard_tokens=17)
    with PackedMemmapDataset(out / "train.bin", 16) as dataset:
        assert len(dataset) == stats.sequences_written
        assert all(length % 16 == 0 for length in dataset.lengths)


def test_missing_label_shards_fail_at_load(tmp_path):
    out, _ = _pack(tmp_path, [{"text": "abcdefghijklmnopqrstuvwxyz"}])
    (out / "train.labels.bin").unlink()
    with pytest.raises(ValueError, match="Missing|missing"):
        PackedMemmapDataset(out / "train.bin", 16)


def test_val_fraction_zero_produces_no_validation_examples(tmp_path):
    out, stats = _pack(tmp_path, [{"text": "abcdefghijklmnopqrstuvwxyz"}])
    assert stats.val_tokens == 0
    assert not (out / "val.bin").exists()


def test_group_split_keeps_long_record_sequences_together(tmp_path):
    path = _source(tmp_path, [{"text": "abcdefghij" * 50}, {"text": "UVWXYZ" * 50}])
    out = tmp_path / "packed"
    build_packed_dataset_streaming([path], out, ByteTokenizer(), seq_len=16,
                                  val_fraction=0.1, cleaning=CleaningConfig(min_chars=1))
    manifest = json.loads((out / "packing.manifest.json").read_text())
    assert manifest["split"]["overlap_groups"] == 0
    train_groups = set(manifest["split"]["train_groups"])
    val_groups = set(manifest["split"]["val_groups"])
    assert train_groups and val_groups
    assert train_groups.isdisjoint(val_groups)


def test_prompt_completion_boundary_masks_prompt(tmp_path):
    path = _source(tmp_path, [{"prompt": "Arithmetic question?", "completion": "Two."}])
    record = next(iter_dataset_records([path]))
    encoded = encode_training_record(record, ByteTokenizer())
    prompt_ids = ByteTokenizer().encode("Arithmetic question?")
    start = next(i for i in range(len(encoded.ids)) if encoded.ids[i:i+len(prompt_ids)] == prompt_ids)
    assert encoded.labels[start:start+len(prompt_ids)] == [-100] * len(prompt_ids)
    assert encoded.loss_masking == "completion_only"


def test_fim_reconstructs_actual_internal_gap():
    transformed = transform_fim_text("abcdefgh")
    prefix, tail = transformed.removeprefix("<|fim_prefix|>").split("<|fim_suffix|>")
    suffix, middle = tail.split("<|fim_middle|>")
    assert prefix + middle + suffix == "abcdefgh"
    assert prefix and suffix and middle


def test_fim_sentinels_are_dedicated_and_roundtrip(tmp_path):
    tokenizer = PurePythonBPETokenizer()
    for sentinel in ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"):
        assert len(tokenizer.encode(sentinel)) == 1
        assert tokenizer.decode(tokenizer.encode(sentinel)) == sentinel
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    assert PurePythonBPETokenizer(path).encode("<|fim_prefix|>") == tokenizer.encode("<|fim_prefix|>")


def test_speaker_text_schema_preserves_roles_and_content(tmp_path):
    path = _source(tmp_path, [{"turns": [
        {"speaker": "user", "text": "Synthetic question?"},
        {"speaker": "assistant", "text": "Synthetic answer."},
    ]}])
    record = next(iter_dataset_records([path]))
    assert "<|assistant|>" in record.text
    assert "Synthetic question?" in record.text
    assert "Synthetic answer." in record.text


@pytest.mark.parametrize("kind", ["bpe", "native-bpe", "sentencepiece", "invalid-kind"])
def test_nonbyte_missing_tokenizer_never_silently_falls_back(kind):
    with pytest.raises(ValueError):
        load_tokenizer(kind)


def test_chat_chunks_include_causal_continuation_history():
    tokenizer = ByteTokenizer()
    record = DatasetRecord(
        "<|user|>\nQ\n<|assistant|>\nABCDEFGHIJKLMN"
        "<|user|>\ncontinue\n<|assistant|>\nOPQRSTUVWXYZ", "synthetic", {"mode": "chat"},
    )
    sequences = encode_training_sequences(record, tokenizer, seq_len=64)
    texts = [tokenizer.decode(sequence.input_ids) for sequence in sequences]
    assert any("ABCDEFGHIJKLMN" in text and "continue" in text for text in texts)


def test_content_fingerprint_is_location_independent(tmp_path):
    first = tmp_path / "one.txt"
    second = tmp_path / "two.txt"
    first.write_text("identical content")
    second.write_text("identical content")
    assert dataset_fingerprint(first) == dataset_fingerprint(second)


def test_gzip_jsonl_is_streamed_as_jsonl(tmp_path):
    path = tmp_path / "source.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"text": "gzip synthetic example"}) + "\n")
    assert next(iter_dataset_records([path])).text.strip() == "gzip synthetic example"


def test_packing_manifest_records_reproducible_identity_and_counts(tmp_path):
    out, stats = _pack(tmp_path, [{"text": "abcdefghijklmnopqrstuvwxyz"}])
    manifest = json.loads((out / "packing.manifest.json").read_text())
    assert manifest["schema_version"] >= 2
    assert manifest["dataset"]["records"] == stats.records_seen == 1
    assert manifest["dataset"]["bytes_read"] > 0
    assert manifest["split"]["seed"] == 1337
    assert manifest["tokenizer"]["fingerprint"]
    assert manifest["cleaning"]["min_chars"] == 1


def test_fixed_chunking_covers_each_target_once():
    sequences = _fixed_shifted_sequences(list(range(10)), list(range(10)), 99,
                                          seq_len=4, mode="generative", loss_masking="none")
    targets = [label for sequence in sequences for label in sequence.labels if label != -100]
    assert targets == list(range(1, 10))


def test_bounded_reader_reports_oversized_structured_record(tmp_path):
    from nanoforge.data.formats import ReaderLimits
    path = _source(tmp_path, [{"text": "x" * 200}])
    stats = DatasetStats()
    assert not list(iter_dataset_records([path], stats=stats,
                                          limits=ReaderLimits(max_record_bytes=64)))
    assert stats.skipped_records == 1
    assert any("limit" in issue.message.lower() for issue in stats.issues)


def test_governance_rejects_pii_and_disallowed_license(tmp_path):
    from nanoforge.data.governance import DataGovernancePolicy, GovernanceStats, govern_records
    records = [
        DatasetRecord("Contact synthetic@example.test", "example", {"license": "MIT"}),
        DatasetRecord("ordinary example", "example", {"license": "proprietary"}),
        DatasetRecord("safe example", "example", {"license": "MIT"}),
    ]
    stats = GovernanceStats()
    kept = list(govern_records(records, DataGovernancePolicy(
        allowed_licenses=("MIT",), pii_action="reject", owner="local-user",
    ), stats))
    assert [record.text for record in kept] == ["safe example"]
    assert stats.pii_rejected == 1 and stats.license_rejected == 1


def test_repository_records_exclude_secrets_and_include_dependencies(tmp_path):
    from nanoforge.data.repository import iter_repository_records
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("from src.b import helper\nprint(helper())\n")
    (tmp_path / "src" / "b.py").write_text("def helper():\n    return 42\n")
    (tmp_path / ".env").write_text("TOKEN=synthetic-secret")
    records = list(iter_repository_records(tmp_path))
    assert len(records) == 2
    record = next(record for record in records if record.source == "src/a.py")
    assert "src/b.py" in record.metadata["dependencies"]
    assert all(".env" not in record.text for record in records)


def test_python_tokenizer_fitting_enforces_memory_budget(tmp_path):
    from nanoforge.data.tokenizer import train_python_bpe_tokenizer
    path = _source(tmp_path, [{"text": "abcdef" * 200}])
    with pytest.raises(ValueError, match="budget|limit"):
        train_python_bpe_tokenizer([path], tmp_path / "tok.json",
                                  max_training_bytes=64, vocab_size=280)


def test_bounded_text_chunks_keep_every_character(tmp_path):
    from nanoforge.data.formats import ReaderLimits
    path = tmp_path / "long.txt"
    text = "abcdefgh" * 80
    path.write_text(text)
    records = list(iter_dataset_records([path], limits=ReaderLimits(max_record_bytes=64)))
    assert "".join(record.text.strip() for record in records) == text
    assert all(len(record.text.encode()) <= 64 for record in records)
    assert len({record.metadata["group_id"] for record in records}) == 1


def test_reader_issue_transcript_is_bounded(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text("invalid\n" * 1100)
    stats = DatasetStats()
    assert not list(iter_dataset_records([path], stats=stats))
    assert stats.invalid_records == stats.issue_count == 1100
    assert len(stats.issues) == 1000 and stats.suppressed_issues == 100


def test_corrupted_shard_fails_integrity_check(tmp_path):
    out, _ = _pack(tmp_path, [{"text": "abcdefghijklmnopqrstuvwxyz"}])
    path = out / "train.bin"
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="integrity"):
        PackedMemmapDataset(path, 16)


def test_failed_split_never_publishes_consumable_partial_data(tmp_path):
    path = _source(tmp_path, [{"text": "a long enough single group" * 3}])
    out = tmp_path / "packed"
    with pytest.raises(ValueError, match="independent"):
        build_packed_dataset_streaming([path], out, ByteTokenizer(), seq_len=16,
                                      val_fraction=0.1, cleaning=CleaningConfig(min_chars=1))
    assert json.loads((out / "packing.manifest.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        PackedMemmapDataset(out / "train.bin", 16)


def test_sampler_resume_restores_future_batches(tmp_path):
    out, _ = _pack(tmp_path, [{"text": "abcdefghijklmnopqrstuvwxyz" * 20}])
    with PackedMemmapDataset(out / "train.bin", 16, seed=7) as dataset:
        state = dataset.state_dict()
        expected = dataset.sample(5)
        dataset.sample(5)
        dataset.load_state_dict(state)
        actual = dataset.sample(5)
        assert all(bool((first == second).all()) for first, second in zip(expected, actual))


def test_legacy_bpe_special_ids_are_preserved(tmp_path):
    specials = list(PurePythonBPETokenizer.special_tokens)
    specials[4:7] = ["<fim_prefix>", "<fim_middle>", "<fim_suffix>"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"type": "nanoforge_python_bpe", "version": 1,
                               "special_tokens": specials, "merges": []}))
    tokenizer = PurePythonBPETokenizer(path)
    assert tokenizer.encode("<fim_prefix>") == [14 + byte for byte in b"<fim_prefix>"]
    assert tokenizer.decode(tokenizer.encode("<fim_prefix>")) == "<fim_prefix>"
    from nanoforge.data.tokenizer import migrate_bpe_tokenizer

    migrated_path = tmp_path / "migrated.json"
    report = migrate_bpe_tokenizer(path, migrated_path)
    assert PurePythonBPETokenizer(migrated_path).encode("<|fim_prefix|>") == [4]
    assert report["source_fingerprint"] != report["destination_fingerprint"]
    assert report["source_version"] == 1 and report["destination_version"] == 2


def test_invalid_bpe_merge_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"type": "nanoforge_python_bpe", "version": 2,
                               "merges": [[99999, 3]]}))
    with pytest.raises(ValueError, match="merge"):
        PurePythonBPETokenizer(path)


def test_native_bpe_and_python_read_identical_artifacts(tmp_path):
    native = pytest.importorskip("nanoforge_tokenizers")
    tokenizer = PurePythonBPETokenizer()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    accelerated = native.ByteLevelBpeTokenizer(str(path))
    text = "alpha <|fim_prefix|>界<|assistant|>omega"
    assert accelerated.encode(text, True, True) == tokenizer.encode(text, True, True)
    assert accelerated.decode(accelerated.encode(text, False, False)) == text


def test_native_cleaning_matches_python_filtering_controls(monkeypatch):
    native = pytest.importorskip("nanoforge_tokenizers")
    from nanoforge.data import cleaning
    texts = ["界界界", "  Ａ\r\nB\0 ", "abcdef1", "abcdef2"]
    config = CleaningConfig(min_chars=1, max_chars=6, near_deduplicate=False)
    monkeypatch.setattr(cleaning, "_load_native_cleaning", lambda: None)
    fallback = [record.text for record in cleaning.clean_records(
        [DatasetRecord(text, "synthetic") for text in texts], config)]
    monkeypatch.setattr(cleaning, "_load_native_cleaning", lambda: native)
    accelerated = [record.text for record in cleaning.clean_records(
        [DatasetRecord(text, "synthetic") for text in texts], config)]
    assert accelerated == fallback


def test_native_fitting_respects_byte_budget(tmp_path):
    native = pytest.importorskip("nanoforge_tokenizers")
    source = tmp_path / "text.txt"
    source.write_text("synthetic training text")
    with pytest.raises(ValueError, match="budget"):
        native.ByteLevelBpeTokenizer.train_from_file(str(source), 280, max_training_bytes=4)


@pytest.mark.parametrize("native", [False, True])
def test_mmap_reader_has_identical_bounded_utf8_behavior(tmp_path, native):
    from nanoforge.data.mmap_reader import MmapCorpusReader
    if native:
        pytest.importorskip("nanoforge_tokenizers")
    path = tmp_path / "corpus.txt"
    path.write_bytes("first 界\nsecond line\n".encode())
    with MmapCorpusReader(path, require_native=native, prefer_native=native,
                          max_record_bytes=32) as reader:
        assert list(reader) == ["first 界\n", "second line\n"]
        assert reader.records == 2 and reader.bytes_read == path.stat().st_size


def test_mmap_reader_rejects_oversized_line(tmp_path):
    from nanoforge.data.mmap_reader import MmapCorpusReader
    path = tmp_path / "long.txt"
    path.write_bytes(b"x" * 64 + b"\nnext\n")
    with MmapCorpusReader(path, prefer_native=False, max_record_bytes=16) as reader:
        with pytest.raises(ValueError, match="limit"):
            next(reader)
        assert next(reader) == "next\n"
        assert reader.skipped_records == 1


def test_code_cleaning_preserves_unicode_literals_and_indentation():
    from nanoforge.data.cleaning import clean_records
    text = "    value = 'Ａ'\n"
    records = list(clean_records([DatasetRecord(text, "snippet.py", {"mode": "code"})],
                                 CleaningConfig(min_chars=1)))
    assert records[0].text == text


def test_custom_group_key_survives_structured_readers(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{"text": "independent first record", "repo_id": "first", "license": "MIT"},
            {"text": "independent second record", "repo_id": "second", "license": "MIT"}]
    path = tmp_path / "grouped.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    records = list(iter_dataset_records([path], group_key="repo_id"))
    assert [record.metadata["repo_id"] for record in records] == ["first", "second"]
    assert all(record.metadata["license"] == "MIT" for record in records)


def test_sqlite_quoted_table_and_completion_schema(tmp_path):
    import sqlite3

    path = tmp_path / "records.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute('create table "order" (prompt text, completion text, group_id text)')
        connection.execute('insert into "order" values (?,?,?)', ("Question?", "Answer.", "synthetic"))
    records = list(iter_dataset_records([path]))
    assert len(records) == 1
    assert records[0].metadata["mode"] == "completion"
    assert records[0].metadata["group_id"] == "synthetic"
    assert "<|assistant|>" in records[0].text


def test_tokenizer_identity_distinguishes_special_encoding_versions():
    current = PurePythonBPETokenizer()
    legacy = PurePythonBPETokenizer()
    legacy.artifact_version = 1
    assert tokenizer_identity(current)["fingerprint"] != tokenizer_identity(legacy)["fingerprint"]


def test_explicit_invalid_fim_does_not_start_a_transaction(tmp_path):
    tokenizer = PurePythonBPETokenizer()
    tokenizer.artifact_version = 1
    out = tmp_path / "packed"
    with pytest.raises(ValueError, match="dedicated"):
        build_packed_dataset_streaming([], out, tokenizer, mode="fim", val_fraction=0)
    assert not out.exists()


def test_auto_fim_rejects_legacy_tokenizer():
    tokenizer = PurePythonBPETokenizer()
    tokenizer.artifact_version = 1
    with pytest.raises(ValueError, match="dedicated"):
        encode_training_record(DatasetRecord("abcdefgh", "synthetic.py", {"mode": "fim"}),
                               tokenizer)


def test_fim_does_not_accept_unknown_token_as_sentinel():
    class UnknownTokenizer:
        bos_id, eos_id, pad_id, unk_id, vocab_size = 1, 2, 0, 3, 8

        def encode(self, text, add_bos=False, add_eos=False):
            return [self.unk_id]

    with pytest.raises(ValueError, match="dedicated"):
        encode_training_record(DatasetRecord("abcdefgh", "synthetic.py"), UnknownTokenizer(),
                               mode="fim")


def test_native_artifact_validation_and_longest_special_match(tmp_path):
    native = pytest.importorskip("nanoforge_tokenizers")
    specials = list(PurePythonBPETokenizer.special_tokens)
    specials[4:7] = ["<custom>", "<custom>suffix", "<fim_suffix>"]
    payload = {"type": "nanoforge_python_bpe", "version": 2,
               "special_tokens": specials, "merges": []}
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(payload))
    python = PurePythonBPETokenizer(path)
    accelerated = native.ByteLevelBpeTokenizer(str(path))
    text = "<custom>suffix <|fim_suffix|>"
    assert accelerated.encode(text, False, False) == python.encode(text)
    assert accelerated.artifact_version == python.artifact_version
    for invalid in ([specials[0]] * 14, [""] + specials[1:]):
        payload["special_tokens"] = invalid
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="layout"):
            native.ByteLevelBpeTokenizer(str(path))
    payload["special_tokens"] = specials
    payload["merges"] = [[14, 15], [14, 15]]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        native.ByteLevelBpeTokenizer(str(path))





