from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Iterator

from nanoforge.data.formats import DatasetRecord
from nanoforge.data.preprocess import stable_hash


@dataclass
class CleaningConfig:
    min_chars: int = 16
    max_chars: int | None = None
    normalize_unicode: bool = True
    collapse_whitespace: bool = False
    deduplicate: bool = True
    near_deduplicate: bool = False
    language: str | None = None


def clean_text(text: str, config: CleaningConfig) -> str:
    text = text.replace("\x00", "")
    if config.normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if config.collapse_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"


def likely_language(text: str) -> str:
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    if ascii_count / max(len(text), 1) > 0.85:
        return "en"
    return "unknown"


def simhash(text: str, bits: int = 64) -> int:
    weights = [0] * bits
    tokens = re.findall(r"\w+", text.lower())
    for token in tokens:
        value = int(stable_hash(token), 16)
        for bit in range(bits):
            weights[bit] += 1 if value & (1 << bit) else -1
    out = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def clean_records(records: Iterable[DatasetRecord], config: CleaningConfig) -> Iterator[DatasetRecord]:
    seen_exact: set[str] = set()
    seen_near: list[int] = []
    for record in records:
        text = clean_text(record.text, config)
        if len(text) < config.min_chars:
            continue
        if config.max_chars is not None and len(text) > config.max_chars:
            text = text[: config.max_chars].rstrip() + "\n"
        if config.language is not None and likely_language(text) != config.language:
            continue
        if config.deduplicate:
            key = stable_hash(text)
            if key in seen_exact:
                continue
            seen_exact.add(key)
        if config.near_deduplicate:
            fp = simhash(text)
            if any(hamming(fp, old) <= 3 for old in seen_near[-50000:]):
                continue
            seen_near.append(fp)
        record.text = text
        yield record

