from __future__ import annotations

import ast
import codecs
import csv
import gzip
import hashlib
import io
import json
import os
import sqlite3
import tarfile
import zipfile
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from nanoforge.data.preprocess import CODE_EXTENSIONS, normalize_text

TEXT_EXTENSIONS = {".txt", ".text", ".md", ".markdown", ".rst", ".log"}
STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".xml", ".sqlite", ".db"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tgz", ".gz", ".tar.gz"}
PRETOKENIZED_EXTENSIONS = {".bin", ".npy", ".npz"}
TEXT_FIELD_HINTS = {
    "text",
    "content",
    "body",
    "prompt",
    "completion",
    "response",
    "answer",
    "instruction",
    "input",
    "output",
    "code",
    "story",
    "messages",
    "conversations",
}


@dataclass
class DatasetRecord:
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetIssue:
    source: str
    message: str
    kind: str = "warning"


@dataclass
class DatasetStats:
    records: int = 0
    bytes_read: int = 0
    skipped_records: int = 0
    invalid_records: int = 0
    issues: list[DatasetIssue] = field(default_factory=list)
    fields: dict[str, int] = field(default_factory=dict)
    formats: dict[str, int] = field(default_factory=dict)
    schemas: dict[str, list[str]] = field(default_factory=dict)
    text_columns: dict[str, list[str]] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    issue_count: int = 0
    suppressed_issues: int = 0

    def add_issue(self, issue: DatasetIssue) -> None:
        self.issue_count += 1
        if len(self.issues) < 1000:
            self.issues.append(issue)
        else:
            self.suppressed_issues += 1

    def add_field(self, name: str) -> None:
        self.fields[name] = self.fields.get(name, 0) + 1

    def add_format(self, name: str) -> None:
        self.formats[name] = self.formats.get(name, 0) + 1

    def add_schema(self, source: str, fields: Iterable[str]) -> None:
        self.schemas[source] = list(fields)

    def add_text_columns(self, source: str, columns: Iterable[str]) -> None:
        self.text_columns[source] = list(columns)


def detect_format(path_or_url: str | Path) -> str:
    raw = str(path_or_url)
    if raw.startswith(("hf://", "hf:")):
        return "huggingface"
    if raw.startswith(("http://", "https://")):
        suffix = Path(raw.split("?", 1)[0]).suffix.lower()
        return "http-jsonl" if suffix == ".jsonl" else "http-text"
    path = Path(raw)
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith(".tar.gz"):
        return "archive"
    if suffix in {".zip", ".tar", ".tgz", ".gz"}:
        return "archive"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".xml":
        return "xml"
    if suffix in {".sqlite", ".db"}:
        return "sqlite"
    if suffix in {".parquet", ".arrow"}:
        return suffix[1:]
    if suffix in PRETOKENIZED_EXTENSIONS:
        return "pretokenized"
    if suffix in TEXT_EXTENSIONS or suffix in CODE_EXTENSIONS:
        return "text"
    return "directory" if path.is_dir() else "text"


@dataclass(frozen=True)
class ReaderLimits:
    """Explicit bounds for one ingestion request; network access is opt-in."""

    max_record_bytes: int = 1_048_576
    max_source_bytes: int = 268_435_456
    max_total_bytes: int = 536_870_912
    max_structured_bytes: int = 8_388_608
    max_archive_members: int = 100_000
    chunk_bytes: int = 65_536
    allow_network: bool = False
    use_mmap: bool = False
    max_files: int = 100_000

    def __post_init__(self):
        for name in ("max_record_bytes", "max_source_bytes", "max_total_bytes",
                     "max_structured_bytes", "max_archive_members", "chunk_bytes", "max_files"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


_READER_LIMITS: ContextVar[ReaderLimits] = ContextVar("nanoforge_reader_limits", default=ReaderLimits())
_GROUP_KEY: ContextVar[str] = ContextVar("nanoforge_dataset_group_key", default="group_id")
_RECORD_METADATA_FIELDS = ("group_id", "document_id", "license", "training_consent")


def iter_dataset_records(
    paths: Iterable[str | Path], *, text_key: str = "text",
    text_columns: Iterable[str] | None = None, code_only: bool = False,
    stats: DatasetStats | None = None, limits: ReaderLimits | None = None,
    group_key: str = "group_id",
) -> Iterator[DatasetRecord]:
    stats = stats if stats is not None else DatasetStats()
    bounds = limits or _READER_LIMITS.get()
    token = _READER_LIMITS.set(bounds)
    group_token = _GROUP_KEY.set(group_key)
    consumed = 0
    try:
        for record in _iter_dataset_records(paths, text_key=text_key, text_columns=text_columns,
                                            code_only=code_only, stats=stats):
            size = len(record.text.encode("utf-8", errors="replace"))
            if size > bounds.max_record_bytes:
                stats.skipped_records += 1
                stats.add_issue(DatasetIssue(record.source, "Record byte limit exceeded"))
                continue
            if consumed + size > bounds.max_total_bytes:
                stats.add_issue(DatasetIssue(record.source, "Total ingestion byte limit exceeded", "error"))
                return
            consumed += size
            stats.records += 1
            stats.bytes_read += size
            yield record
    finally:
        _READER_LIMITS.reset(token)
        _GROUP_KEY.reset(group_token)

def _iter_dataset_records(
    paths: Iterable[str | Path],
    *,
    text_key: str = "text",
    text_columns: Iterable[str] | None = None,
    code_only: bool = False,
    stats: DatasetStats | None = None,
) -> Iterator[DatasetRecord]:
    stats = stats or DatasetStats()
    configured_columns = tuple(text_columns or ())
    for raw in paths:
        fmt = detect_format(raw)
        stats.add_format(fmt)
        bounds = _READER_LIMITS.get()
        if str(raw).startswith(("http://", "https://", "hf://", "hf:")) and not bounds.allow_network:
            stats.add_issue(DatasetIssue(str(raw), "Network ingestion requires allow_network capability", "error"))
            continue
        if not str(raw).startswith(("http://", "https://", "hf://", "hf:")):
            path = Path(raw)
            if path.exists():
                if path.is_file() and path.stat().st_size > bounds.max_source_bytes:
                    stats.add_issue(DatasetIssue(str(raw), "Source byte limit exceeded", "error"))
                    continue
                if fmt in {"json", "yaml", "xml"} and path.stat().st_size > bounds.max_structured_bytes:
                    stats.add_issue(DatasetIssue(str(raw), "Structured document byte limit exceeded; use JSONL", "error"))
                    continue
                stats.fingerprints[str(path)] = dataset_fingerprint(path)
        if fmt == "directory":
            yield from _iter_directory(
                Path(raw),
                text_key=text_key,
                text_columns=configured_columns,
                code_only=code_only,
                stats=stats,
            )
        elif fmt == "archive":
            yield from _iter_archive(
                Path(raw),
                text_key=text_key,
                text_columns=configured_columns,
                code_only=code_only,
                stats=stats,
            )
        elif fmt == "jsonl":
            yield from _iter_jsonl(Path(raw), text_key=text_key, text_columns=configured_columns, stats=stats)
        elif fmt == "json":
            yield from _iter_json(Path(raw), text_key=text_key, text_columns=configured_columns, stats=stats)
        elif fmt == "csv":
            yield from _iter_csv(Path(raw), text_key=text_key, text_columns=configured_columns, stats=stats)
        elif fmt == "yaml":
            yield from _iter_yaml(Path(raw), text_key=text_key, text_columns=configured_columns, stats=stats)
        elif fmt == "xml":
            yield from _iter_xml(Path(raw), stats=stats)
        elif fmt == "sqlite":
            yield from _iter_sqlite(Path(raw), text_key=text_key, text_columns=configured_columns, stats=stats)
        elif fmt in {"parquet", "arrow"}:
            yield from _iter_arrow_like(
                Path(raw), fmt=fmt, text_key=text_key, text_columns=configured_columns, stats=stats
            )
        elif fmt == "huggingface":
            yield from _iter_huggingface(
                str(raw), text_key=text_key, text_columns=configured_columns, stats=stats
            )
        elif fmt.startswith("http"):
            yield from _iter_http(str(raw), text_key=text_key, text_columns=configured_columns, stats=stats)
        elif fmt == "pretokenized":
            stats.add_issue(DatasetIssue(str(raw), "Pretokenized files are consumed by training directly."))
        else:
            yield from _iter_text_file(Path(raw), stats=stats)


def inspect_dataset(paths: Iterable[str | Path], text_key: str = "text", limit: int = 1000) -> DatasetStats:
    stats = DatasetStats()
    for idx, record in enumerate(iter_dataset_records(paths, text_key=text_key, stats=stats)):
        for key in record.metadata:
            stats.add_field(key)
        if idx + 1 >= limit:
            break
    return stats


def bounded_source_files(root: Path, *, max_files: int = 100000,
                         excluded_directories: tuple[str, ...] = (".git", ".venv", "node_modules", "__pycache__")) -> Iterator[Path]:
    """Deterministic traversal with an explicit bound and directory pruning."""
    stack = [root]
    inspected = 0
    excluded = {name.casefold() for name in excluded_directories}
    while stack:
        directory = stack.pop()
        entries = []
        with os.scandir(directory) as iterator:
            for entry in iterator:
                inspected += 1
                if inspected > max_files:
                    raise ValueError("Source file-count limit exceeded")
                if entry.is_symlink() or entry.name.casefold() in excluded:
                    continue
                entries.append(entry)
        for entry in sorted(entries, key=lambda item: item.name, reverse=True):
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield Path(entry.path)

def dataset_fingerprint(path: Path) -> str:
    """Stream a SHA-256 content identity independent of the source's location.

    Directory identities include relative names and each file's content digest.
    Hashing reads full contents in bounded chunks; it is not a metadata shortcut.
    """

    digest = hashlib.sha256()
    if path.is_dir():
        for file in bounded_source_files(path, max_files=_READER_LIMITS.get().max_files):
            digest.update(file.relative_to(path).as_posix().encode("utf-8"))
            digest.update(bytes.fromhex(dataset_fingerprint(file)))
    else:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def sanitize_text(value: Any) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    text = text.replace("\x00", "")
    # Unicode normalization belongs to the explicit cleaning policy; doing it
    # here would irreversibly alter code literals before code-preserving cleaning.
    return normalize_text(text)


def _iter_directory(
    path: Path,
    text_key: str,
    text_columns: Iterable[str],
    code_only: bool,
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    for file in bounded_source_files(path, max_files=_READER_LIMITS.get().max_files):
        if file.is_file() and not file.is_symlink() and not any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in file.relative_to(path).parts):
            if code_only and file.suffix.lower() not in CODE_EXTENSIONS:
                continue
            yield from _iter_dataset_records(
                [file], text_key=text_key, text_columns=text_columns, code_only=code_only, stats=stats
            )


def _decode_bytes(data: bytes, source: str, stats: DatasetStats) -> str | None:
    if b"\x00" in data[:4096] and not source.lower().endswith((".utf16", ".utf-16")):
        stats.add_issue(DatasetIssue(source, "Binary-looking file skipped.", "warning"))
        stats.skipped_records += 1
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        stats.invalid_records += 1
        stats.add_issue(DatasetIssue(source, "Invalid UTF-8 recovered with replacement characters."))
        try:
            return data.decode("utf-8", errors="replace")
        except Exception as exc:
            stats.add_issue(DatasetIssue(source, f"Could not decode text: {exc}", "error"))
            return None


def _iter_text_stream(stream, source: str, stats: DatasetStats) -> Iterator[DatasetRecord]:
    bounds = _READER_LIMITS.get()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending: list[str] = []
    pending_bytes = 0
    total = 0
    chunk_index = 0
    record_budget = max(1, bounds.max_record_bytes - 4)
    fingerprint = stats.fingerprints.get(source, hashlib.sha256(source.split("!")[-1].encode()).hexdigest())
    while True:
        data = stream.read(min(bounds.chunk_bytes, max(1, record_budget - pending_bytes)))
        if not data:
            tail = decoder.decode(b"", final=True)
            if tail:
                pending.append(tail)
            if pending:
                yield DatasetRecord(normalize_text("".join(pending)), source, {
                    "format": "text", "suffix": Path(source).suffix,
                    "group_id": fingerprint, "chunk": chunk_index,
                })
            return
        total += len(data)
        if total > bounds.max_source_bytes:
            stats.add_issue(DatasetIssue(source, "Decompressed source byte limit exceeded", "error"))
            return
        if total == len(data) and b"\x00" in data[:4096]:
            stats.skipped_records += 1
            stats.add_issue(DatasetIssue(source, "Binary-looking file skipped"))
            return
        buffered = decoder.getstate()[0]
        try:
            (buffered + data).decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data":
                stats.invalid_records += 1
                stats.add_issue(DatasetIssue(source, "Invalid UTF-8 recovered with replacement characters"))
        part = decoder.decode(data)
        pending.append(part)
        pending_bytes += len(part.encode("utf-8"))
        if pending_bytes >= record_budget:
            yield DatasetRecord(normalize_text("".join(pending)), source, {
                "format": "text", "suffix": Path(source).suffix,
                "group_id": fingerprint, "chunk": chunk_index,
            })
            chunk_index += 1
            pending = []
            pending_bytes = 0


def _iter_text_file(path: Path, stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        bounds = _READER_LIMITS.get()
        if bounds.use_mmap:
            from nanoforge.data.mmap_reader import MmapCorpusReader

            fingerprint = stats.fingerprints.get(str(path)) or dataset_fingerprint(path)
            with MmapCorpusReader(path, max_record_bytes=bounds.max_record_bytes,
                                  max_source_bytes=bounds.max_source_bytes) as reader:
                index = 0
                invalid_seen = 0
                while True:
                    try:
                        text = next(reader)
                    except StopIteration:
                        break
                    except ValueError as exc:
                        stats.skipped_records += 1
                        stats.add_issue(DatasetIssue(str(path), str(exc)))
                        continue
                    if reader.invalid_utf8_records > invalid_seen:
                        stats.invalid_records += reader.invalid_utf8_records - invalid_seen
                        invalid_seen = reader.invalid_utf8_records
                        stats.add_issue(DatasetIssue(str(path), "Invalid UTF-8 recovered with replacement characters"))
                    yield DatasetRecord(normalize_text(text), str(path), {
                        "format": "text", "suffix": path.suffix,
                        "group_id": fingerprint, "line": index + 1,
                        "reader_backend": reader.backend,
                    })
                    index += 1
            return
        with path.open("rb") as stream:
            yield from _iter_text_stream(stream, str(path), stats)
    except (OSError, ValueError) as exc:
        stats.add_issue(DatasetIssue(str(path), f"Could not read file: {exc}", "error"))


def _extract_text_from_row(
    row: Any,
    text_key: str,
    text_columns: Iterable[str] | None = None,
) -> str | None:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        configured = [key for key in text_columns or () if key in row and row.get(key) is not None]
        if configured:
            parts = []
            for key in configured:
                val = row[key]
                if val is None:
                    continue
                if key in {"messages", "conversations"}:
                    formatted = _format_messages(val)
                    if formatted.strip():
                        parts.append(formatted)
                else:
                    cell = _coerce_cell(val)
                    if cell.strip():
                        parts.append(cell)
            return "\n".join(parts) if parts else None
        if text_key in row and text_key in {"messages", "conversations"}:
            return _format_messages(row[text_key])
        if text_key in row:
            return _coerce_cell(row[text_key])
        if "messages" in row:
            return _format_messages(row["messages"])
        if "conversations" in row:
            return _format_messages(row["conversations"])
        if "instruction" in row and any(key in row for key in ("input", "output", "response", "answer")):
            prompt = "\n".join(_coerce_cell(part) for part in (row.get("instruction"), row.get("input")) if part)
            response = _coerce_cell(row.get("output", row.get("response", row.get("answer"))))
            return f"<|user|>\n{prompt}\n<|assistant|>\n{response}" if response else prompt
        if "prompt" in row and any(key in row for key in ("completion", "response", "answer")):
            response = next((_coerce_cell(row[key]) for key in ("completion", "response", "answer") if row.get(key) is not None), "")
            return f"<|user|>\n{_coerce_cell(row['prompt'])}\n<|assistant|>\n{response}"
        turn_field = _best_turn_field(row)
        if turn_field is not None:
            turns = row[turn_field]
            formatted = (
                _format_alternating(turns)
                if all(isinstance(v, str) for v in turns)
                else _format_messages(turns)
            )
            if formatted.strip():
                return formatted
        key = _best_text_key(row, text_key)
        if key is not None:
            return _coerce_cell(row[key])
    return None


def _coerce_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _best_text_key(row: dict[str, Any], text_key: str) -> str | None:
    best: tuple[int, str] | None = None
    for key, value in row.items():
        if key in (*_RECORD_METADATA_FIELDS, _GROUP_KEY.get()) and key != text_key:
            continue
        if value is None or isinstance(value, (int, float, bool)):
            continue
        text = _coerce_cell(value)
        if not text.strip():
            continue
        score = min(len(text), 4096)
        lowered = key.lower()
        if lowered == text_key.lower():
            score += 100_000
        if lowered in TEXT_FIELD_HINTS:
            score += 50_000
        if any(hint in lowered for hint in TEXT_FIELD_HINTS):
            score += 10_000
        candidate = (score, key)
        if best is None or candidate > best:
            best = candidate
    return best[1] if best else None


def _is_role_dict(item: Any) -> bool:
    """A message-like dict, e.g. {"role": "user", "content": "..."} or ShareGPT {"from","value"}."""
    if not isinstance(item, dict):
        return False
    has_role = any(k in item for k in ("role", "from", "speaker"))
    has_text = any(k in item for k in ("content", "value", "text", "message"))
    return has_role and has_text


def _looks_like_turn_list(value: Any) -> bool:
    """True for a conversation held in a bare list — either alternating strings
    (UltraChat ``{"id","data":[...]}``) or a list of role dicts (ShareGPT/OpenAI)."""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    strings = [v for v in value if isinstance(v, str) and v.strip()]
    if len(strings) == len(value) and len(strings) >= 2:
        # A list of non-trivial strings; guard against short tag/label lists.
        return sum(len(s) for s in strings) >= 40
    return all(_is_role_dict(v) for v in value)


def _best_turn_field(row: dict[str, Any]) -> str | None:
    """Pick the list-valued field that best represents a conversation."""
    best: tuple[int, str] | None = None
    for key, value in row.items():
        if not _looks_like_turn_list(value):
            continue
        size = sum(len(_coerce_cell(v)) for v in value)
        candidate = (size, str(key))
        if best is None or candidate > best:
            best = candidate
    return best[1] if best else None


def _infer_row_mode(row: Any) -> str:
    if not isinstance(row, dict):
        return "generative"
    keys = {str(key).lower() for key in row}
    if "messages" in keys or "conversations" in keys:
        return "chat"
    if "instruction" in keys and ({"output", "response", "answer"} & keys):
        return "instruct"
    if "prompt" in keys and ({"completion", "response", "answer"} & keys):
        return "completion"
    if "code" in keys:
        return "code"
    # Value-based fallback: a conversation may live in an arbitrarily named list
    # field (UltraChat "data", "turns", "dialog", ...). Detect by shape, not name.
    if _best_turn_field(row) is not None:
        return "chat"
    return "generative"


def _format_alternating(turns: Any) -> str:
    """Format a bare list of strings as alternating user/assistant turns.
    Even index = user, odd index = assistant (UltraChat convention)."""
    parts = []
    for idx, turn in enumerate(turns):
        text = _coerce_cell(turn).strip()
        if not text:
            continue
        role = "user" if idx % 2 == 0 else "assistant"
        parts.append(f"<|{role}|>\n{text}")
    return "\n".join(parts)


def _format_messages(messages: Any) -> str:
    if isinstance(messages, str):
        stripped = messages.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                messages = json.loads(stripped)
            except Exception:
                try:
                    messages = ast.literal_eval(stripped)
                except Exception:
                    return stripped
    parts = []
    if isinstance(messages, dict):
        messages = messages.get("messages", messages.get("conversations", [messages]))
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", msg.get("speaker", "user")))
                if role == "human":
                    role = "user"
                if role == "gpt":
                    role = "assistant"
                content = msg.get("content", msg.get("value", msg.get("text", msg.get("message", ""))))
                parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def _row_metadata(record_row: Any, **metadata) -> dict[str, Any]:
    if isinstance(record_row, dict):
        for name in (*_RECORD_METADATA_FIELDS, _GROUP_KEY.get()):
            if name in record_row:
                metadata[name] = record_row[name]
        if "group_id" not in metadata and "document_id" in metadata:
            metadata["group_id"] = metadata["document_id"]
    return metadata


def _iter_jsonl_stream(stream, source: str, text_key: str,
                       text_columns: Iterable[str], stats: DatasetStats) -> Iterator[DatasetRecord]:
    bounds = _READER_LIMITS.get()
    total = 0
    line_no = 0
    while True:
        line = stream.readline(bounds.max_record_bytes + 1)
        if not line:
            return
        line_no += 1
        total += len(line)
        if total > bounds.max_source_bytes:
            stats.add_issue(DatasetIssue(source, "Decompressed source byte limit exceeded", "error"))
            return
        if len(line) > bounds.max_record_bytes:
            while line and not line.endswith(b"\n"):
                line = stream.readline(bounds.max_record_bytes + 1)
                total += len(line)
                if total > bounds.max_source_bytes:
                    stats.add_issue(DatasetIssue(source, "Source byte limit exceeded", "error"))
                    return
            stats.skipped_records += 1
            stats.add_issue(DatasetIssue(source, f"Record byte limit exceeded at line {line_no}"))
            continue
        if not line.strip():
            continue
        try:
            decoded = line.decode("utf-8")
        except UnicodeDecodeError:
            decoded = line.decode("utf-8", errors="replace")
            stats.invalid_records += 1
            stats.add_issue(DatasetIssue(source, f"Invalid UTF-8 recovered at line {line_no}"))
        try:
            row = json.loads(decoded)
        except json.JSONDecodeError as exc:
            stats.invalid_records += 1
            stats.add_issue(DatasetIssue(source, f"Bad JSONL line {line_no}: {exc}"))
            continue
        if isinstance(row, dict):
            stats.add_schema(source, row.keys())
            for key in row:
                stats.add_field(key)
        text = _extract_text_from_row(row, text_key, text_columns)
        if text:
            yield DatasetRecord(sanitize_text(text), source, _row_metadata(
                row, format="jsonl", line=line_no, mode=_infer_row_mode(row),
            ))
        else:
            stats.skipped_records += 1


def _iter_jsonl(path: Path, text_key: str, text_columns: Iterable[str],
                stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        with path.open("rb") as stream:
            yield from _iter_jsonl_stream(stream, str(path), text_key, text_columns, stats)
    except (OSError, ValueError) as exc:
        stats.add_issue(DatasetIssue(str(path), f"Could not read JSONL: {exc}", "error"))


def _iter_json(
    path: Path,
    text_key: str,
    text_columns: Iterable[str],
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        stats.add_issue(DatasetIssue(str(path), f"Could not read JSON: {exc}", "error"))
        return
    rows = data if isinstance(data, list) else data.get("data", [data]) if isinstance(data, dict) else [data]
    for idx, row in enumerate(rows):
        if isinstance(row, dict):
            stats.add_schema(str(path), row.keys())
            for key in row:
                stats.add_field(key)
        text = _extract_text_from_row(row, text_key, text_columns)
        if text:
            yield DatasetRecord(
                sanitize_text(text),
                str(path),
                _row_metadata(row, format="json", index=idx, mode=_infer_row_mode(row)),
            )
        else:
            stats.skipped_records += 1


def _iter_csv(path: Path, text_key: str, text_columns: Iterable[str], stats: DatasetStats) -> Iterator[DatasetRecord]:
    dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, dialect=dialect)
            if reader.fieldnames:
                stats.add_schema(str(path), reader.fieldnames)
            for idx, row in enumerate(reader):
                for key in row:
                    stats.add_field(key)
                text = _extract_text_from_row(row, text_key, text_columns)
                if text is None:
                    text = "\n".join(str(v) for v in row.values() if v)
                yield DatasetRecord(
                    sanitize_text(text),
                    str(path),
                    _row_metadata(row, format="csv", row=idx, mode=_infer_row_mode(row)),
                )
    except Exception as exc:
        stats.add_issue(DatasetIssue(str(path), f"Could not read CSV/TSV: {exc}", "error"))


def _iter_yaml(path: Path, text_key: str, text_columns: Iterable[str], stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        import yaml
    except Exception:
        yield from _iter_text_file(path, stats)
        return
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        stats.add_issue(DatasetIssue(str(path), f"Could not read YAML: {exc}", "error"))
        return
    rows = data if isinstance(data, list) else [data]
    for idx, row in enumerate(rows):
        if isinstance(row, dict):
            stats.add_schema(str(path), row.keys())
            for key in row:
                stats.add_field(key)
        text = _extract_text_from_row(row, text_key, text_columns)
        if text:
            yield DatasetRecord(
                sanitize_text(text),
                str(path),
                _row_metadata(row, format="yaml", index=idx, mode=_infer_row_mode(row)),
            )


def _iter_xml(path: Path, stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        root = ElementTree.parse(path).getroot()
    except Exception as exc:
        stats.add_issue(DatasetIssue(str(path), f"Could not read XML: {exc}", "error"))
        return
    for idx, node in enumerate(root.iter()):
        text = " ".join(t.strip() for t in node.itertext() if t.strip())
        if text:
            yield DatasetRecord(normalize_text(text), str(path), {"format": "xml", "index": idx, "tag": node.tag})


def _iter_sqlite(path: Path, text_key: str, text_columns: Iterable[str],
                 stats: DatasetStats) -> Iterator[DatasetRecord]:
    connection = None

    def quote(identifier):
        return '"' + identifier.replace('"', '""') + '"'

    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        tables = connection.execute("select name from sqlite_master where type='table'").fetchall()
        for (table,) in tables:
            cols = [entry[1] for entry in connection.execute(f"pragma table_info({quote(table)})")]
            stats.add_schema(f"{path}:{table}", cols)
            selected = _select_columns(cols, text_key, text_columns)
            if not selected:
                continue
            stats.add_text_columns(f"{path}:{table}", selected)
            metadata_columns = [name for name in (*_RECORD_METADATA_FIELDS, _GROUP_KEY.get())
                                if name in cols and name not in selected]
            queried = selected + metadata_columns
            columns = ", ".join(quote(name) for name in queried)
            for index, values in enumerate(connection.execute(f"select {columns} from {quote(table)}")):
                row = dict(zip(queried, values))
                text = _extract_text_from_row(row, text_key, text_columns)
                if text:
                    yield DatasetRecord(sanitize_text(text), str(path), _row_metadata(
                        row, format="sqlite", table=table, row=index, mode=_infer_row_mode(row),
                    ))
                else:
                    stats.skipped_records += 1
    except (OSError, sqlite3.Error, ValueError) as exc:
        stats.add_issue(DatasetIssue(str(path), f"Could not read SQLite: {exc}", "error"))
    finally:
        if connection is not None:
            connection.close()


def _select_columns(
    names: Iterable[str],
    text_key: str,
    text_columns: Iterable[str] | None = None,
) -> list[str]:
    names = list(names)
    configured = [name for name in text_columns or () if name in names]
    if configured:
        return configured
    if text_key in names:
        return [text_key]
    hinted = [name for name in names if name.lower() in TEXT_FIELD_HINTS]
    if hinted:
        return hinted[:4]
    fuzzy = [name for name in names if any(hint in name.lower() for hint in TEXT_FIELD_HINTS)]
    return fuzzy[:4]


def _iter_arrow_like(
    path: Path,
    fmt: str,
    text_key: str,
    text_columns: Iterable[str],
    stats: DatasetStats,
    batch_size: int = 1024,
) -> Iterator[DatasetRecord]:
    try:
        import pyarrow.ipc as ipc
        import pyarrow.parquet as pq
    except Exception:
        stats.add_issue(DatasetIssue(str(path), f"Install pyarrow to read {fmt} datasets."))
        return
    try:
        if fmt == "parquet":
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
            selected = _select_columns(names, text_key, text_columns)
            stats.add_schema(str(path), names)
            stats.add_text_columns(str(path), selected)
            if not selected:
                stats.add_issue(DatasetIssue(str(path), "No usable text column found."))
                return
            metadata_columns = [name for name in (*_RECORD_METADATA_FIELDS, _GROUP_KEY.get()) if name in names and name not in selected]
            for batch_id, batch in enumerate(parquet.iter_batches(batch_size=batch_size, columns=selected + metadata_columns)):
                yield from _records_from_arrow_batch(batch, str(path), fmt, batch_id, text_key, text_columns, stats)
            return
        with path.open("rb") as fh:
            try:
                reader = ipc.open_file(fh)
                schema = reader.schema
                batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
            except Exception:
                fh.seek(0)
                reader = ipc.open_stream(fh)
                schema = reader.schema
                batches = iter(reader)
            names = schema.names
            selected = _select_columns(names, text_key, text_columns)
            stats.add_schema(str(path), names)
            stats.add_text_columns(str(path), selected)
            if not selected:
                stats.add_issue(DatasetIssue(str(path), "No usable text column found."))
                return
            metadata_columns = [name for name in (*_RECORD_METADATA_FIELDS, _GROUP_KEY.get()) if name in names and name not in selected]
            for batch_id, batch in enumerate(batches):
                yield from _records_from_arrow_batch(
                    batch.select(selected + metadata_columns), str(path), fmt, batch_id, text_key, text_columns, stats
                )
    except Exception as exc:
        stats.add_issue(DatasetIssue(str(path), f"Could not read {fmt}: {exc}", "error"))


def _records_from_arrow_batch(
    batch: Any,
    source: str,
    fmt: str,
    batch_id: int,
    text_key: str,
    text_columns: Iterable[str],
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    for row_id, row in enumerate(batch.to_pylist()):
        if isinstance(row, dict):
            for key in row:
                stats.add_field(key)
        text = _extract_text_from_row(row, text_key, text_columns)
        if text:
            yield DatasetRecord(
                sanitize_text(text),
                source,
                _row_metadata(row, format=fmt, batch=batch_id, row=row_id, mode=_infer_row_mode(row)),
            )
        else:
            stats.skipped_records += 1


def _records_from_stream(stream, source: str, text_key: str,
                         text_columns: Iterable[str], stats: DatasetStats) -> Iterator[DatasetRecord]:
    name = urlsplit(source).path if source.startswith(("http://", "https://")) else source
    suffix = Path(name).suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_stream(stream, source, text_key, text_columns, stats)
    elif suffix == ".json":
        bounds = _READER_LIMITS.get()
        data = stream.read(bounds.max_structured_bytes + 1)
        if len(data) > bounds.max_structured_bytes:
            stats.add_issue(DatasetIssue(source, "Structured document byte limit exceeded; use JSONL", "error"))
            return
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            stats.invalid_records += 1
            stats.add_issue(DatasetIssue(source, f"Invalid JSON: {exc}"))
            return
        rows = payload if isinstance(payload, list) else payload.get("data", [payload]) if isinstance(payload, dict) else [payload]
        for index, row in enumerate(rows):
            text = _extract_text_from_row(row, text_key, text_columns)
            if text:
                yield DatasetRecord(sanitize_text(text), source, _row_metadata(
                    row, format="json", index=index, mode=_infer_row_mode(row),
                ))
            else:
                stats.skipped_records += 1
    else:
        yield from _iter_text_stream(stream, source, stats)


def _iter_archive(path: Path, text_key: str, text_columns: Iterable[str],
                  code_only: bool, stats: DatasetStats) -> Iterator[DatasetRecord]:
    bounds = _READER_LIMITS.get()
    try:
        if path.suffix.lower() == ".gz" and not path.name.lower().endswith(".tar.gz"):
            with gzip.open(path, "rb") as stream:
                yield from _records_from_stream(stream, str(path.with_suffix("")), text_key, text_columns, stats)
            return
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > bounds.max_archive_members:
                    stats.add_issue(DatasetIssue(str(path), "Archive member count limit exceeded", "error"))
                    return
                for info in infos:
                    suffix = Path(info.filename).suffix.lower()
                    if info.is_dir() or (code_only and suffix not in CODE_EXTENSIONS):
                        continue
                    if suffix not in TEXT_EXTENSIONS | CODE_EXTENSIONS | STRUCTURED_EXTENSIONS:
                        continue
                    if info.file_size > bounds.max_source_bytes:
                        stats.add_issue(DatasetIssue(str(path), "Archive member byte limit exceeded", "error"))
                        continue
                    with archive.open(info) as stream:
                        yield from _records_from_stream(stream, f"{path}!{info.filename}", text_key, text_columns, stats)
        else:
            with tarfile.open(path, mode="r|*") as archive:
                for index, member in enumerate(archive):
                    if index >= bounds.max_archive_members:
                        stats.add_issue(DatasetIssue(str(path), "Archive member count limit exceeded", "error"))
                        return
                    suffix = Path(member.name).suffix.lower()
                    if not member.isfile() or (code_only and suffix not in CODE_EXTENSIONS):
                        continue
                    if suffix not in TEXT_EXTENSIONS | CODE_EXTENSIONS | STRUCTURED_EXTENSIONS:
                        continue
                    if member.size > bounds.max_source_bytes:
                        stats.add_issue(DatasetIssue(str(path), "Archive member byte limit exceeded", "error"))
                        continue
                    stream = archive.extractfile(member)
                    if stream is not None:
                        with stream:
                            yield from _records_from_stream(stream, f"{path}!{member.name}", text_key, text_columns, stats)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        stats.add_issue(DatasetIssue(str(path), f"Could not read archive: {exc}", "error"))


def _records_from_archive_bytes(data: bytes, source: str, text_key: str,
                               text_columns: Iterable[str], stats: DatasetStats) -> Iterator[DatasetRecord]:
    # Compatibility helper for callers holding an already bounded byte record.
    yield from _records_from_stream(io.BytesIO(data), source, text_key, text_columns, stats)


def _iter_http(url: str, text_key: str, text_columns: Iterable[str],
               stats: DatasetStats) -> Iterator[DatasetRecord]:
    if not _READER_LIMITS.get().allow_network:
        stats.add_issue(DatasetIssue(url, "Network ingestion requires allow_network capability", "error"))
        return
    try:
        request = Request(url, headers={"User-Agent": "nanoforge/0.1"})
        with urlopen(request, timeout=30) as response:
            yield from _records_from_stream(response, url, text_key, text_columns, stats)
    except (OSError, ValueError) as exc:
        stats.add_issue(DatasetIssue(url, f"Could not read HTTP dataset: {exc}", "error"))


def _iter_huggingface(
    ref: str,
    text_key: str,
    text_columns: Iterable[str],
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    try:
        from datasets import load_dataset
    except Exception:
        stats.add_issue(DatasetIssue(ref, "Install datasets to stream Hugging Face datasets."))
        return
    name = ref.removeprefix("hf://").removeprefix("hf:")
    if not name:
        stats.add_issue(DatasetIssue(ref, "Hugging Face dataset reference is empty.", "error"))
        return
    try:
        dataset = load_dataset(name, split="train", streaming=True)
        for idx, row in enumerate(dataset):
            if isinstance(row, dict):
                for key in row:
                    stats.add_field(key)
            text = _extract_text_from_row(row, text_key, text_columns)
            if text:
                yield DatasetRecord(
                    sanitize_text(text),
                    ref,
                    _row_metadata(row, format="huggingface", row=idx, mode=_infer_row_mode(row)),
                )
            else:
                stats.skipped_records += 1
    except Exception as exc:
        stats.add_issue(DatasetIssue(ref, f"Could not stream Hugging Face dataset: {exc}", "error"))
