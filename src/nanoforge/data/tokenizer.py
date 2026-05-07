from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol
import tempfile
import pandas as pd

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
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


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


TokenizerAdapter = ByteTokenizer | TokenizersBPE | WordPieceTokenizer | SentencePieceTokenizer


def load_tokenizer(tokenizer_type: str = "byte", path: str | Path | None = None) -> TokenizerAdapter:
    if tokenizer_type == "byte" or path is None:
        return ByteTokenizer()
    if tokenizer_type == "bpe":
        return TokenizersBPE(path)
    if tokenizer_type == "wordpiece":
        return WordPieceTokenizer(path)
    if tokenizer_type in {"sp", "sentencepiece"}:
        return SentencePieceTokenizer(path)
    raise ValueError(f"Unknown tokenizer_type={tokenizer_type}")


def train_bpe_tokenizer(
    files: Iterable[str | Path],
    out_path: str | Path,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> None:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    except Exception as exc:
        raise RuntimeError("Install tokenizers to train BPE tokenizers: pip install tokenizers") from exc

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
   

    processed_files = []

    for p in files:
        p = Path(p)

        # parquet support
        if p.suffix.lower() == ".parquet":
            try:
                df = pd.read_parquet(p)

                candidate_cols = [
                    "text",
                    "content",
                    "prompt",
                    "completion",
                    "story",
                    "code",
                    "messages"
                ]

                text_col = None

                for col in candidate_cols:
                    if col in df.columns:
                        text_col = col
                        break



                if text_col is None:
                    print(f"SKIP parquet no text column: {p}")
                    continue

                tmp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".txt",
                    mode="w",
                    encoding="utf-8"
                )

                for x in df[text_col]:
                    if isinstance(x, str):
                        tmp.write(x + "\n")

                tmp.close()

                processed_files.append(tmp.name)

                print(f"Loaded parquet: {p}")

            except Exception as e:
                print(f"Failed parquet {p}: {e}")

        else:
            processed_files.append(str(p))

    tokenizer.train(processed_files, trainer=trainer)


    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))


def train_wordpiece_tokenizer(
    files: Iterable[str | Path],
    out_path: str | Path,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> None:
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
    tokenizer.train([str(p) for p in files], trainer=trainer)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))


def train_sentencepiece_tokenizer(
    files: Iterable[str | Path],
    out_prefix: str | Path,
    vocab_size: int = 32000,
    model_type: str = "bpe",
) -> None:
    try:
        import sentencepiece as spm
    except Exception as exc:
        raise RuntimeError("Install sentencepiece to train SentencePiece tokenizers") from exc
    input_arg = ",".join(str(p) for p in files)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=input_arg,
        model_prefix=str(out_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols="<fim_prefix>,<fim_middle>,<fim_suffix>",
    )
