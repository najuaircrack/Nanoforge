"""Generate a reproducible demo chat dataset from raw UltraChat.

Streams ``data/raw/train_1.jsonl`` (the UltraChat ``{"id","data":[turn, ...]}``
shape), sanitizes each turn's text (mojibake / control-char cleanup via the same
``sanitize_text`` the loaders use), takes a *deterministic* first-N subset (by line
order, no RNG — so re-running produces the identical file), and writes the subset
back out in the original schema so it exercises the structure-aware profiler
end to end.

Two intended sizes:
  * ``--n 2500``  -> the small committed sample ``data/demo/chat.jsonl`` (a few MB)
  * ``--n 30000`` -> the (uncommitted, regenerable) set used to train the demo model

Usage:
    python scripts/make_demo_dataset.py --n 2500 --out data/demo/chat.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanoforge.data.formats import sanitize_text  # noqa: E402


def _clean_turns(turns: list) -> list[str] | None:
    """Sanitize each string turn; drop conversations that don't look usable."""
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    cleaned: list[str] = []
    for turn in turns:
        if not isinstance(turn, str):
            return None
        text = sanitize_text(turn).strip()
        if not text:
            return None
        cleaned.append(text)
    return cleaned


def make_demo_dataset(source: Path, out: Path, n: int) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open("r", encoding="utf-8", errors="replace") as fh, \
            out.open("w", encoding="utf-8") as of:
        for line in fh:
            if written >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            turns = _clean_turns(row.get("data"))
            if turns is None:
                continue
            record = {"id": str(row.get("id", written)), "data": turns}
            of.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reproducible demo chat dataset.")
    parser.add_argument("--source", default="data/raw/train_1.jsonl", help="Raw UltraChat JSONL.")
    parser.add_argument("--out", default="data/demo/chat.jsonl", help="Output JSONL path.")
    parser.add_argument("--n", type=int, default=2500, help="Number of conversations to keep.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")
    written = make_demo_dataset(source, Path(args.out), args.n)
    print(f"wrote {written} conversations to {args.out}")


if __name__ == "__main__":
    main()
