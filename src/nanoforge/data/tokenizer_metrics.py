from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from nanoforge.data.formats import iter_dataset_records
from nanoforge.data.tokenizer import TokenizerLike


@dataclass
class TokenizerReport:
    documents: int
    characters: int
    bytes_utf8: int
    tokens: int
    chars_per_token: float
    bytes_per_token: float
    unique_tokens: int
    unk_rate: float
    top_tokens: list[tuple[int, int]]


def evaluate_tokenizer(
    tokenizer: TokenizerLike,
    paths: Iterable[str | Path],
    *,
    text_key: str = "text",
    limit: int = 1000,
) -> TokenizerReport:
    counts: Counter[int] = Counter()
    documents = 0
    characters = 0
    bytes_utf8 = 0
    tokens_total = 0
    unk_total = 0
    for record in iter_dataset_records(paths, text_key=text_key):
        ids = tokenizer.encode(record.text)
        documents += 1
        characters += len(record.text)
        bytes_utf8 += len(record.text.encode("utf-8", errors="ignore"))
        tokens_total += len(ids)
        unk_total += sum(1 for token in ids if token == tokenizer.unk_id)
        counts.update(ids)
        if documents >= limit:
            break
    denom = max(tokens_total, 1)
    return TokenizerReport(
        documents=documents,
        characters=characters,
        bytes_utf8=bytes_utf8,
        tokens=tokens_total,
        chars_per_token=characters / denom,
        bytes_per_token=bytes_utf8 / denom,
        unique_tokens=len(counts),
        unk_rate=unk_total / denom,
        top_tokens=counts.most_common(50),
    )


def save_tokenizer_report(report: TokenizerReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

