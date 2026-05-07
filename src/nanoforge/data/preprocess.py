from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

CODE_EXTENSIONS = {
    ".py",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".rs",
    ".pwn",
    ".inc",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"


def stable_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()


def iter_text_files(paths: Iterable[str | Path], code_only: bool = False) -> Iterator[tuple[Path, str]]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files = (p for p in path.rglob("*") if p.is_file())
        else:
            files = [path]
        for file in files:
            if code_only and file.suffix.lower() not in CODE_EXTENSIONS:
                continue
            try:
                text = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    text = file.read_text(encoding="latin-1")
                except Exception:
                    continue
            except Exception:
                continue
            yield file, normalize_text(text)


def iter_jsonl(path: str | Path, text_key: str = "text") -> Iterator[str]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if text_key in row:
                yield normalize_text(str(row[text_key]))
            elif {"messages"} <= set(row):
                parts = []
                for msg in row["messages"]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    parts.append(f"<|{role}|>\n{content}")
                yield normalize_text("\n".join(parts))


def deduplicate(records: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    seen: set[str] = set()
    for name, text in records:
        key = stable_hash(text)
        if key in seen:
            continue
        seen.add(key)
        yield name, text

