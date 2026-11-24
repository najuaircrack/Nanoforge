from __future__ import annotations

import json
import re
import tracemalloc
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from nanoforge.data.formats import DatasetStats, ReaderLimits, iter_dataset_records
from nanoforge.data.native_tokenizer import (
    NativeBPETokenizer,
    NativeByteTokenizer,
    train_native_bpe_from_texts,
)


class TokenizerLike(Protocol):
    bos_id: int
    eos_id: int
    pad_id: int
    unk_id: int
    vocab_size: int

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """Dependency-free byte tokenizer for smoke tests and tiny local experiments."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    offset = 4
    vocab_size = 260

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [b + self.offset for b in text.encode("utf-8", errors="replace")]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        data = bytes([i - self.offset for i in ids if self.offset <= i < self.offset + 256])
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "byte"}), encoding="utf-8")


class TokenizersBPE:
    def __init__(self, path: str | Path):
        if _looks_like_builtin_bpe(path):
            self._fallback = PurePythonBPETokenizer(path)
            self.path = Path(path)
            self.pad_id = self._fallback.pad_id
            self.bos_id = self._fallback.bos_id
            self.eos_id = self._fallback.eos_id
            self.unk_id = self._fallback.unk_id
            self.vocab_size = self._fallback.vocab_size
            return
        self._fallback = None
        try:
            from tokenizers import Tokenizer
        except Exception as exc:
            raise RuntimeError("Install tokenizers to use BPE tokenizers: pip install tokenizers") from exc
        self.path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path))
        vocab = self.tokenizer.get_vocab()
        self.pad_id = vocab.get("<pad>", 0)
        self.bos_id = vocab.get("<bos>", 1)
        self.eos_id = vocab.get("<eos>", 2)
        self.unk_id = vocab.get("<unk>", 3)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if self._fallback is not None:
            return self._fallback.encode(text, add_bos=add_bos, add_eos=add_eos)
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        if self._fallback is not None:
            return self._fallback.decode(ids)
        return self.tokenizer.decode(ids)


class PurePythonBPETokenizer:
    """Small byte-level BPE fallback for local/native-free environments."""

    special_tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|pad|>",
        "<|bos|>",
        "<|eos|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|endoftext|>",
    ]

    def __init__(self, path: str | Path | None = None, merges: list[tuple[int, int]] | None = None):
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.byte_offset = len(self.special_tokens)
        self.artifact_version = 2
        self.merges = merges or []
        if path is not None:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if raw.get("type") != "nanoforge_python_bpe":
                raise ValueError(f"{path} is not a Nanoforge Python BPE tokenizer.")
            if raw.get("version", 1) not in {1, 2}:
                raise ValueError("Unsupported tokenizer artifact schema version")
            self.artifact_version = raw.get("version", 1)
            self.merges = [tuple(pair) for pair in raw.get("merges", [])]
            saved_special = raw.get("special_tokens", self.special_tokens)
            if len(saved_special) != self.byte_offset or any(not isinstance(token, str) or not token for token in saved_special) or len(set(saved_special)) != self.byte_offset:
                raise ValueError("Tokenizer special-token layout requires an explicit migration")
            self.special_tokens = list(saved_special)
        self.special_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}
        # Migrated v2 layouts can retain older FIM names at their saved IDs.
        # V1 encoding still treats every literal as bytes until explicit migration.
        for name in ("prefix", "middle", "suffix"):
            old = f"<fim_{name}>"
            if old in self.special_to_id:
                self.special_to_id[f"<|fim_{name}|>"] = self.special_to_id[old]
        self._special_pattern = re.compile("(" + "|".join(re.escape(token) for token in sorted(self.special_to_id, key=len, reverse=True)) + ")")
        self.pair_to_id: dict[tuple[int, int], int] = {}
        for idx, pair in enumerate(self.merges):
            if len(pair) != 2 or any(not isinstance(token, int) or token < self.byte_offset or token >= self.byte_offset + 256 + idx for token in pair):
                raise ValueError("Tokenizer contains an invalid merge reference")
            if pair in self.pair_to_id:
                raise ValueError("Tokenizer contains duplicate merge pairs")
            self.pair_to_id[pair] = self.byte_offset + 256 + idx
        self.id_to_piece: dict[int, tuple[int, ...]] = {
            self.byte_offset + byte: (byte,) for byte in range(256)
        }
        for idx, pair in enumerate(self.merges):
            token_id = self.byte_offset + 256 + idx
            left = self.id_to_piece.get(pair[0], ())
            right = self.id_to_piece.get(pair[1], ())
            self.id_to_piece[token_id] = left + right
        self.vocab_size = self.byte_offset + 256 + len(self.merges)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = []
        parts = self._special_pattern.split(text) if self.artifact_version >= 2 else [text]
        for part in parts:
            if self.artifact_version >= 2 and part in self.special_to_id:
                ids.append(self.special_to_id[part])
                continue
            pieces = [self.byte_offset + byte for byte in part.encode("utf-8", errors="replace")]
            for pair, merged_id in self.pair_to_id.items():
                if len(pieces) < 2:
                    break
                pieces = _merge_pair_once(pieces, pair, merged_id)
            ids.extend(pieces)
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        data: list[int] = []
        for token_id in ids:
            if token_id in self.id_to_piece:
                data.extend(self.id_to_piece[token_id])
            elif self.artifact_version >= 2 and 4 <= token_id < self.byte_offset:
                data.extend(self.special_tokens[token_id].encode("utf-8"))
        return bytes(data).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "nanoforge_python_bpe",
            "version": self.artifact_version,
            "special_tokens": self.special_tokens,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def migrate_bpe_tokenizer(source: str | Path, destination: str | Path) -> dict[str, object]:
    """Create a v2 artifact explicitly; existing checkpoints require adaptation.

    Version 1 encoded special literals as bytes. Changing this semantic contract
    never happens on load and never overwrites the original tokenizer.
    """
    import hashlib

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Tokenizer migration requires a separate output artifact")
    tokenizer = PurePythonBPETokenizer(source)
    migrated = PurePythonBPETokenizer(merges=list(tokenizer.merges))
    migrated.save(destination)
    return {
        "source_version": tokenizer.artifact_version, "destination_version": 2,
        "source_fingerprint": hashlib.sha256(source.read_bytes()).hexdigest(),
        "destination_fingerprint": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "semantic_change": "special literals become dedicated token IDs",
        "checkpoint_action": "retrain or adapt before using the migrated tokenizer",
    }

class WordPieceTokenizer:
    def __init__(self, path: str | Path):
        try:
            from tokenizers import Tokenizer
        except Exception as exc:
            raise RuntimeError("Install tokenizers to use WordPiece tokenizers: pip install tokenizers") from exc
        self.path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path))
        vocab = self.tokenizer.get_vocab()
        self.pad_id = vocab.get("[PAD]", vocab.get("<pad>", 0))
        self.bos_id = vocab.get("<bos>", 1)
        self.eos_id = vocab.get("<eos>", 2)
        self.unk_id = vocab.get("[UNK]", vocab.get("<unk>", 3))
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


class SentencePieceTokenizer:
    def __init__(self, path: str | Path):
        try:
            import sentencepiece as spm
        except Exception as exc:
            raise RuntimeError("Install sentencepiece to use SentencePiece: pip install sentencepiece") from exc
        self.path = Path(path)
        self.processor = spm.SentencePieceProcessor(model_file=str(path))
        self.pad_id = self.processor.pad_id() if self.processor.pad_id() >= 0 else 0
        self.bos_id = self.processor.bos_id() if self.processor.bos_id() >= 0 else 1
        self.eos_id = self.processor.eos_id() if self.processor.eos_id() >= 0 else 2
        self.unk_id = self.processor.unk_id() if self.processor.unk_id() >= 0 else 3
        self.vocab_size = self.processor.vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode(ids)


TokenizerAdapter = ByteTokenizer | NativeByteTokenizer | NativeBPETokenizer | TokenizersBPE | PurePythonBPETokenizer | WordPieceTokenizer | SentencePieceTokenizer


@dataclass
class TokenizerTrainingReport:
    records: int = 0
    chars: int = 0
    skipped_records: int = 0
    invalid_records: int = 0
    peak_memory_mb: float = 0.0
    dry_run: bool = False
    training_bytes: int = 0
    fit_seconds: float = 0.0
    fitting_peak_memory_mb: float = 0.0
    peak_memory_kind: str = "python-allocator"
    backend: str = "inspection-only"
    budget_bytes: int | None = None


def iter_tokenizer_training_texts(
    files: Iterable[str | Path],
    *,
    text_key: str = "text",
    text_columns: Iterable[str] | None = None,
    max_records: int | None = None,
    stats: DatasetStats | None = None,
    reader_limits: ReaderLimits | None = None,
) -> Iterable[str]:
    count = 0
    for record in iter_dataset_records(files, text_key=text_key, text_columns=text_columns, stats=stats, limits=reader_limits):
        if max_records is not None and count >= max_records:
            break
        count += 1
        yield record.text


def load_tokenizer(tokenizer_type: str = "byte", path: str | Path | None = None) -> TokenizerAdapter:
    tokenizer_type = tokenizer_type.strip().lower().replace("_", "-")
    if tokenizer_type in {"byte-native", "native-byte"}:
        return NativeByteTokenizer()
    if tokenizer_type in {"byte-native-required", "native-byte-required"}:
        return NativeByteTokenizer(require_native=True)
    if tokenizer_type == "byte":
        return ByteTokenizer()
    known = {"native-bpe", "bpe-native", "native-bpe-required", "bpe-native-required", "python-bpe", "pure-python-bpe", "bpe", "wordpiece", "sp", "sentencepiece", "unigram"}
    if tokenizer_type not in known:
        raise ValueError(f"Unknown tokenizer_type={tokenizer_type}")
    if path is None:
        raise ValueError(f"tokenizer_type={tokenizer_type} requires a tokenizer artifact path")
    if tokenizer_type in {"native-bpe", "bpe-native"}:
        return NativeBPETokenizer(path)
    if tokenizer_type in {"native-bpe-required", "bpe-native-required"}:
        return NativeBPETokenizer(path, require_native=True)
    if tokenizer_type in {"python-bpe", "pure-python-bpe"}:
        if path is None:
            return PurePythonBPETokenizer()
        return PurePythonBPETokenizer(path)
    if tokenizer_type == "bpe":
        return TokenizersBPE(path)
    if tokenizer_type == "wordpiece":
        return WordPieceTokenizer(path)
    if tokenizer_type in {"sp", "sentencepiece", "unigram"}:
        return SentencePieceTokenizer(path)
    raise ValueError(f"Unknown tokenizer_type={tokenizer_type}")


def train_bpe_tokenizer(
    files: Iterable[str | Path],
    out_path: str | Path,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    *,
    text_key: str = "text",
    text_columns: Iterable[str] | None = None,
    dry_run: bool = False,
    max_records: int | None = None,
    max_training_bytes: int = 16_777_216,
    reader_limits: ReaderLimits | None = None,
) -> TokenizerTrainingReport:
    files = list(files)
    stats = DatasetStats()
    report = _scan_training_texts(files, text_key, text_columns, max_records, stats, reader_limits)
    report.dry_run = dry_run
    if dry_run:
        return report
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    except Exception:
        return train_python_bpe_tokenizer(
            files,
            out_path,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            text_key=text_key,
            text_columns=text_columns,
            dry_run=False,
            max_records=max_records,
            scanned_report=report,
            max_training_bytes=max_training_bytes,
            reader_limits=reader_limits,
        )

    special = PurePythonBPETokenizer.special_tokens
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special,
        show_progress=True,
    )

    iterator = iter_tokenizer_training_texts(
        files,
        text_key=text_key,
        text_columns=text_columns,
        max_records=max_records,
        stats=DatasetStats(),
        reader_limits=reader_limits,
    )
    with _fit_measurement(report, "huggingface-bpe", max_training_bytes):
        tokenizer.train_from_iterator(_bounded_texts(iterator, report, max_training_bytes), trainer=trainer)


    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return report


def train_python_bpe_tokenizer(
    files: Iterable[str | Path],
    out_path: str | Path,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    *,
    text_key: str = "text",
    text_columns: Iterable[str] | None = None,
    dry_run: bool = False,
    max_records: int | None = None,
    scanned_report: TokenizerTrainingReport | None = None,
    max_training_bytes: int = 16_777_216,
    reader_limits: ReaderLimits | None = None,
) -> TokenizerTrainingReport:
    files = list(files)
    stats = DatasetStats()
    report = scanned_report or _scan_training_texts(files, text_key, text_columns, max_records, stats, reader_limits)
    report.dry_run = dry_run
    if dry_run:
        return report
    import time
    from array import array

    if max_training_bytes < 1:
        raise ValueError("Tokenizer training byte budget must be positive")
    if vocab_size < len(PurePythonBPETokenizer.special_tokens) + 256:
        raise ValueError("Byte-level BPE vocabulary must include all byte and special tokens")
    token_offset = len(PurePythonBPETokenizer.special_tokens)
    words: list[array] = []
    total_bytes = 0
    started = time.perf_counter()
    tracing_owned = not tracemalloc.is_tracing()
    if tracing_owned:
        tracemalloc.start()
    try:
        for text in iter_tokenizer_training_texts(files, text_key=text_key,
                                                  text_columns=text_columns, max_records=max_records, reader_limits=reader_limits):
            raw = text.encode("utf-8", errors="replace")
            total_bytes += len(raw)
            if total_bytes > max_training_bytes:
                raise ValueError(f"Tokenizer training byte budget exceeded ({max_training_bytes}); increase the explicit budget or use a native/external fitter")
            if raw:
                words.append(array("I", (token_offset + byte for byte in raw)))
        merges: list[tuple[int, int]] = []
        if not words:
            raise ValueError("No usable records for tokenizer training")
        next_id = token_offset + 256
        for _ in range(max(0, vocab_size - next_id)):
            counts: Counter[tuple[int, int]] = Counter()
            for word in words:
                counts.update(zip(word, word[1:]))
            if not counts:
                break
            pair, count = counts.most_common(1)[0]
            if count < min_frequency:
                break
            merges.append(pair)
            words = [array("I", _merge_pair_once(word, pair, next_id)) for word in words]
            next_id += 1
        PurePythonBPETokenizer(merges=merges).save(out_path)
        report.training_bytes = total_bytes
        report.fit_seconds = time.perf_counter() - started
        report.fitting_peak_memory_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        report.peak_memory_mb = max(report.peak_memory_mb, report.fitting_peak_memory_mb)
        report.backend = "python-array-bpe"
        report.budget_bytes = max_training_bytes
        return report
    finally:
        if tracing_owned:
            tracemalloc.stop()


def train_native_bpe_tokenizer(
    files: Iterable[str | Path],
    out_path: str | Path,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    *,
    text_key: str = "text",
    text_columns: Iterable[str] | None = None,
    dry_run: bool = False,
    max_records: int | None = None,
    require_native: bool = False,
    show_progress: bool = True,
    max_training_bytes: int = 16_777_216,
    reader_limits: ReaderLimits | None = None,
) -> TokenizerTrainingReport:
    files = list(files)
    stats = DatasetStats()
    report = _scan_training_texts(files, text_key, text_columns, max_records, stats, reader_limits)
    report.dry_run = dry_run
    if dry_run:
        return report
    texts = iter_tokenizer_training_texts(
        files,
        text_key=text_key,
        text_columns=text_columns,
        max_records=max_records,
        stats=DatasetStats(),
        reader_limits=reader_limits,
    )
    with _fit_measurement(report, "rust-bpe-python-allocator-only", max_training_bytes):
        if train_native_bpe_from_texts(
            _bounded_texts(texts, report, max_training_bytes), out_path,
            vocab_size=vocab_size, min_frequency=min_frequency,
            require_native=require_native, show_progress=show_progress,
            max_training_bytes=max_training_bytes,
        ):
            return report
    return train_python_bpe_tokenizer(
        files,
        out_path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        text_key=text_key,
        text_columns=text_columns,
        dry_run=False,
        max_records=max_records,
        scanned_report=report,
        max_training_bytes=max_training_bytes,
        reader_limits=reader_limits,
    )


def _merge_pair_once(ids: list[int], pair: tuple[int, int], merged_id: int) -> list[int]:
    out: list[int] = []
    idx = 0
    while idx < len(ids):
        if idx + 1 < len(ids) and ids[idx] == pair[0] and ids[idx + 1] == pair[1]:
            out.append(merged_id)
            idx += 2
        else:
            out.append(ids[idx])
            idx += 1
    return out


def _looks_like_builtin_bpe(path: str | Path) -> bool:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            prefix = handle.read(128)
        return '"type"' in prefix and "nanoforge_python_bpe" in prefix
    except OSError:
        return False


def train_wordpiece_tokenizer(
    files: Iterable[str | Path],
    out_path: str | Path,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    *,
    text_key: str = "text",
    text_columns: Iterable[str] | None = None,
    dry_run: bool = False,
    max_records: int | None = None,
    max_training_bytes: int = 16_777_216,
    reader_limits: ReaderLimits | None = None,
) -> TokenizerTrainingReport:
    files = list(files)
    stats = DatasetStats()
    report = _scan_training_texts(files, text_key, text_columns, max_records, stats, reader_limits)
    report.dry_run = dry_run
    if dry_run:
        return report
    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    except Exception as exc:
        raise RuntimeError("Install tokenizers to train WordPiece tokenizers: pip install tokenizers") from exc

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<bos>", "<eos>"]
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special,
        show_progress=True,
    )
    iterator = iter_tokenizer_training_texts(
        files,
        text_key=text_key,
        text_columns=text_columns,
        max_records=max_records,
        stats=DatasetStats(),
        reader_limits=reader_limits,
    )
    with _fit_measurement(report, "huggingface-wordpiece", max_training_bytes):
        tokenizer.train_from_iterator(_bounded_texts(iterator, report, max_training_bytes), trainer=trainer)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return report


def train_sentencepiece_tokenizer(
    files: Iterable[str | Path],
    out_prefix: str | Path,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    *,
    text_key: str = "text",
    text_columns: Iterable[str] | None = None,
    dry_run: bool = False,
    max_records: int | None = None,
    max_training_bytes: int = 16_777_216,
    reader_limits: ReaderLimits | None = None,
) -> TokenizerTrainingReport:
    files = list(files)
    stats = DatasetStats()
    report = _scan_training_texts(files, text_key, text_columns, max_records, stats, reader_limits)
    report.dry_run = dry_run
    if dry_run:
        return report
    try:
        import sentencepiece as spm
    except Exception as exc:
        raise RuntimeError("Install sentencepiece to train SentencePiece tokenizers") from exc
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with _fit_measurement(report, "sentencepiece-python-allocator-only", max_training_bytes):
        spm.SentencePieceTrainer.Train(
            sentence_iterator=_bounded_texts(iter_tokenizer_training_texts(
                files,
                text_key=text_key,
                text_columns=text_columns,
                max_records=max_records,
                stats=DatasetStats(),
                reader_limits=reader_limits,
            ), report, max_training_bytes),
            model_prefix=str(out_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            user_defined_symbols="<|fim_prefix|>,<|fim_middle|>,<|fim_suffix|>",
        )
    return report


def _bounded_texts(texts: Iterable[str], report: TokenizerTrainingReport, budget: int):
    if budget < 1:
        raise ValueError("Tokenizer training byte budget must be positive")
    report.training_bytes = 0
    for text in texts:
        report.training_bytes += len(text.encode("utf-8", errors="replace"))
        if report.training_bytes > budget:
            raise ValueError("Tokenizer training byte budget exceeded")
        yield text


@contextmanager
def _fit_measurement(report: TokenizerTrainingReport, backend: str, budget: int):
    import time

    tracing_owned = not tracemalloc.is_tracing()
    if tracing_owned:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        yield
    finally:
        report.fit_seconds = time.perf_counter() - started
        report.fitting_peak_memory_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        report.peak_memory_mb = max(report.peak_memory_mb, report.fitting_peak_memory_mb)
        report.backend = backend
        report.budget_bytes = budget
        if tracing_owned:
            tracemalloc.stop()

def _scan_training_texts(
    files: Iterable[str | Path],
    text_key: str,
    text_columns: Iterable[str] | None,
    max_records: int | None,
    stats: DatasetStats,
    reader_limits: ReaderLimits | None = None,
) -> TokenizerTrainingReport:
    tracing_owned = not tracemalloc.is_tracing()
    if tracing_owned:
        tracemalloc.start()
    records = 0
    chars = 0
    try:
        for text in iter_tokenizer_training_texts(
            files,
            text_key=text_key,
            text_columns=text_columns,
            max_records=max_records,
            stats=stats,
            reader_limits=reader_limits,
        ):
            records += 1
            chars += len(text)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        if tracing_owned:
            tracemalloc.stop()
    return TokenizerTrainingReport(
        records=records,
        chars=chars,
        skipped_records=stats.skipped_records,
        invalid_records=stats.invalid_records,
        peak_memory_mb=peak / (1024 * 1024),
    )
