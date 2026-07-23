"""Structure-aware dataset profiling.

Given any dataset (local files, archives, HTTP, HuggingFace refs) this inspects a
sample of records and *understands its structure by value, not just field name* —
so nonstandard online shapes like UltraChat ``{"id","data":[turn, turn, ...]}`` are
recognized as chat conversations and trained with assistant-only loss masking.

The result drives ``auto-train --mode auto``: detected mode, the text column(s) to
train on, and the loss-masking strategy. Nothing here loads a full corpus into RAM;
it streams and stops after ``limit`` records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from nanoforge.data.formats import (
    DatasetStats,
    _best_turn_field,
    _is_role_dict,
    detect_format,
    iter_dataset_records,
)

_MASKING_FOR_MODE = {
    "chat": "assistant_only",
    "instruct": "completion_only",
    "completion": "completion_only",
    "code": "none",
    "generative": "none",
}


@dataclass
class DatasetProfile:
    """What the profiler understood about a dataset."""

    detected_mode: str = "generative"
    recommended_text_columns: tuple[str, ...] = ()
    recommended_loss_masking: str = "none"
    turn_field: str | None = None
    role_style: str = "flat-text"  # alternating-list | role-dicts | flat-text
    confidence: float = 0.0
    num_sampled: int = 0
    mode_votes: dict[str, int] = field(default_factory=dict)
    fields: dict[str, int] = field(default_factory=dict)
    sample_texts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_mode": self.detected_mode,
            "recommended_text_columns": list(self.recommended_text_columns),
            "recommended_loss_masking": self.recommended_loss_masking,
            "turn_field": self.turn_field,
            "role_style": self.role_style,
            "confidence": round(self.confidence, 4),
            "num_sampled": self.num_sampled,
            "mode_votes": self.mode_votes,
            "fields": self.fields,
            "sample_texts": self.sample_texts,
            "notes": self.notes,
        }

    def summary_line(self) -> str:
        via = f" via {self.role_style} field '{self.turn_field}'" if self.turn_field else ""
        return (
            f"detected: {self.detected_mode}{via}, "
            f"masking={self.recommended_loss_masking}, "
            f"confidence={self.confidence:.2f} (n={self.num_sampled})"
        )


def _sample_raw_rows(paths: Iterable[str | Path], limit: int) -> Iterator[dict[str, Any]]:
    """Best-effort sample of raw dict rows for row-based formats (jsonl/json).

    Used only to detect the *turn field* and *role style* for the human-readable
    summary; the authoritative mode vote comes from the streaming reader below, so
    formats without a cheap raw view simply skip structural detail.
    """
    seen = 0
    for raw in paths:
        if seen >= limit:
            return
        fmt = detect_format(raw)
        path = Path(raw)
        if fmt == "jsonl" and path.exists():
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if seen >= limit:
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        seen += 1
                        yield row
        elif fmt == "json" and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                if seen >= limit:
                    return
                if isinstance(row, dict):
                    seen += 1
                    yield row


def _role_style_for(row: dict[str, Any], turn_field: str | None) -> str:
    if turn_field is None:
        return "flat-text"
    value = row.get(turn_field)
    if isinstance(value, (list, tuple)) and value:
        if all(_is_role_dict(v) for v in value):
            return "role-dicts"
        if all(isinstance(v, str) for v in value):
            return "alternating-list"
    return "flat-text"


def profile_dataset(
    paths: Iterable[str | Path],
    *,
    text_key: str = "text",
    limit: int = 200,
) -> DatasetProfile:
    """Inspect up to ``limit`` records and infer how to train on them."""

    path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
    profile = DatasetProfile()

    # 1. Authoritative mode vote via the streaming reader. Each record's metadata
    #    already carries the (value-aware) inferred mode, across every format.
    stats = DatasetStats()
    votes: dict[str, int] = {}
    samples: list[str] = []
    n = 0
    for record in iter_dataset_records(path_list, text_key=text_key, stats=stats):
        mode = str(record.metadata.get("mode", "generative"))
        votes[mode] = votes.get(mode, 0) + 1
        if len(samples) < 3 and record.text.strip():
            samples.append(record.text[:400])
        n += 1
        if n >= limit:
            break

    profile.num_sampled = n
    profile.mode_votes = votes
    profile.fields = dict(stats.fields)
    profile.sample_texts = samples

    if not votes:
        profile.notes.append("No usable records found; defaulting to generative.")
        return profile

    detected, top = max(votes.items(), key=lambda kv: kv[1])
    profile.detected_mode = detected
    profile.confidence = top / max(1, n)
    profile.recommended_loss_masking = _MASKING_FOR_MODE.get(detected, "none")

    # 2. Structural detail (turn field / role style) from a cheap raw sample.
    turn_field: str | None = None
    role_style = "flat-text"
    for row in _sample_raw_rows(path_list, limit=min(limit, 50)):
        candidate = _best_turn_field(row)
        if candidate is not None:
            turn_field = candidate
            role_style = _role_style_for(row, candidate)
            break
    profile.turn_field = turn_field
    profile.role_style = role_style

    # 3. Recommended text columns.
    if turn_field is not None:
        profile.recommended_text_columns = (turn_field,)
    elif detected == "chat":
        for named in ("messages", "conversations"):
            if named in profile.fields:
                profile.recommended_text_columns = (named,)
                break

    if detected == "chat" and turn_field and role_style == "alternating-list":
        profile.notes.append(
            f"Alternating user/assistant turns in list field '{turn_field}' "
            "(UltraChat style); even index=user, odd index=assistant."
        )
    if profile.confidence < 0.6:
        profile.notes.append(
            "Mixed record shapes; consider passing --mode/--text-column explicitly."
        )
    return profile
