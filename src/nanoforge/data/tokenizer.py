from __future__ import annotations

import json
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from nanoforge.data.formats import DatasetStats, iter_dataset_records
from nanoforge.data.native_tokenizer import NativeBPETokenizer, NativeByteTokenizer, train_native_bpe_from_texts

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
        "<fim_prefix>",
        "<fim_middle>",
        "<fim_suffix>",
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
        self.merges = merges or []
        if path is not None:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if raw.get("type") != "nanoforge_python_bpe":
                raise ValueError(f"{path} is not a Nanoforge Python BPE tokenizer.")
            self.merges = [tuple(pair) for pair in raw.get("merges", [])]
        self.pair_to_id: dict[tuple[int, int], int] = {}
        for idx, pair in enumerate(self.merges):
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
        ids = [self.byte_offset + byte for byte in text.encode("utf-8", errors="replace")]
        for pair, merged_id in self.pair_to_id.items():
            if len(ids) < 2:
                break
            ids = _merge_pair_once(ids, pair, merged_id)
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
        return bytes(data).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "nanoforge_python_bpe",
            "version": 1,
            "special_tokens": self.special_tokens,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


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


def iter_tokenizer_training_texts(
    files: Iterable[str | Path],
    *,
    text_key: str = "text",
    text_columns: Iterable[str] | None = None,
    max_records: int | None = None,
    stats: DatasetStats | None = None,
) -> Iterable[str]:
    count = 0
    for record in iter_dataset_records(files, text_key=text_key, text_columns=text_columns, stats=stats):
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
    if tokenizer_type == "byte" or path is None:
        return ByteTokenizer()
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
) -> TokenizerTrainingReport:
    files = list(files)
    stats = DatasetStats()
    report = _scan_training_texts(files, text_key, text_columns, max_records, stats)
    report.dry_run = dry_run
    if dry_run:
        return report
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    except Exception as exc:
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
        )

    special = ["<pad>", "<bos>", "<eos>", "<unk>", "<fim_prefix>", "<fim_middle>", "<fim_suffix>", "<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>"]
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
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)


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
) -> TokenizerTrainingReport:
    files = list(files)
    stats = DatasetStats()
    report = scanned_report or _scan_training_texts(files, text_key, text_columns, max_records, stats)
    report.dry_run = dry_run
    if dry_run:
        return report
    token_offset = len(PurePythonBPETokenizer.special_tokens)
    words: list[list[int]] = []
    for text in iter_tokenizer_training_texts(files, text_key=text_key, text_columns=text_columns, max_records=max_records):
        encoded = [token_offset + byte for byte in text.encode("utf-8", errors="replace")]
        if encoded:
            words.append(encoded)
    merges: list[tuple[int, int]] = []
    next_id = token_offset + 256
    target_merges = max(0, vocab_size - next_id)
    for _ in range(target_merges):
        counts: Counter[tuple[int, int]] = Counter()
        for word in words:
            counts.update(zip(word, word[1:]))
        if not counts:
            break
        pair, count = counts.most_common(1)[0]
        if count < min_frequency:
            break
        merges.append(pair)
        words = [_merge_pair_once(word, pair, next_id) for word in words]
        next_id += 1
    PurePythonBPETokenizer(merges=merges).save(out_path)
    return report


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
) -> TokenizerTrainingReport:
    files = list(files)
    stats = DatasetStats()
    report = _scan_training_texts(files, text_key, text_columns, max_records, stats)
    report.dry_run = dry_run
    if dry_run:
        return report
    texts = iter_tokenizer_training_texts(
        files,
        text_key=text_key,
        text_columns=text_columns,
        max_records=max_records,
        stats=DatasetStats(),
    )
    if train_native_bpe_from_texts(
        texts,
        out_path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        require_native=require_native,
        show_progress=show_progress,
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
) -> TokenizerTrainingReport:
    files = list(files)
    stats = DatasetStats()
    report = _scan_training_texts(files, text_key, text_columns, max_records, stats)
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
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)
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
) -> TokenizerTrainingReport:
    files = list(files)
    stats = DatasetStats()
    report = _scan_training_texts(files, text_key, text_columns, max_records, stats)
    report.dry_run = dry_run
    if dry_run:
        return report
    try:
        import sentencepiece as spm
    except Exception as exc:
        raise RuntimeError("Install sentencepiece to train SentencePiece tokenizers") from exc
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        sentence_iterator=iter_tokenizer_training_texts(
            files,
            text_key=text_key,
            text_columns=text_columns,
            max_records=max_records,
            stats=DatasetStats(),
        ),
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols="<fim_prefix>,<fim_middle>,<fim_suffix>",
    )
    return report


def _scan_training_texts(
    files: Iterable[str | Path],
    text_key: str,
    text_columns: Iterable[str] | None,
    max_records: int | None,
    stats: DatasetStats,
) -> TokenizerTrainingReport:
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
        ):
            records += 1
            chars += len(text)
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return TokenizerTrainingReport(
        records=records,
        chars=chars,
        skipped_records=stats.skipped_records,
        invalid_records=stats.invalid_records,
        peak_memory_mb=peak / (1024 * 1024),
    )
