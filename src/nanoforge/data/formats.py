from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import tarfile
import unicodedata
import zipfile
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator
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


def iter_dataset_records(
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
        if not str(raw).startswith(("http://", "https://", "hf://", "hf:")):
            path = Path(raw)
            if path.exists():
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
            stats.issues.append(DatasetIssue(str(raw), "Pretokenized files are consumed by training directly."))
        else:
            yield from _iter_text_file(Path(raw), stats=stats)


def inspect_dataset(paths: Iterable[str | Path], text_key: str = "text", limit: int = 1000) -> DatasetStats:
    stats = DatasetStats()
    for idx, record in enumerate(iter_dataset_records(paths, text_key=text_key, stats=stats)):
        stats.records += 1
        stats.bytes_read += len(record.text.encode("utf-8", errors="ignore"))
        for key in record.metadata:
            stats.add_field(key)
        if idx + 1 >= limit:
            break
    return stats


def dataset_fingerprint(path: Path) -> str:
    """Cheap, deterministic fingerprint for duplicate dataset detection.

    This avoids hashing entire multi-GB corpora during inspection. Preprocessing manifests can
    combine it with chunk-level hashes when a full content fingerprint is required.
    """

    stat = path.stat()
    payload = f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime_ns)}"
    return hashlib.blake2b(payload.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()


def sanitize_text(value: Any) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFKC", text)
    return normalize_text(text)


def _iter_directory(
    path: Path,
    text_key: str,
    text_columns: Iterable[str],
    code_only: bool,
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    for file in path.rglob("*"):
        if file.is_file():
            if code_only and file.suffix.lower() not in CODE_EXTENSIONS:
                continue
            yield from iter_dataset_records(
                [file], text_key=text_key, text_columns=text_columns, code_only=code_only, stats=stats
            )


def _decode_bytes(data: bytes, source: str, stats: DatasetStats) -> str | None:
    if b"\x00" in data[:4096] and not source.lower().endswith((".utf16", ".utf-16")):
        stats.issues.append(DatasetIssue(source, "Binary-looking file skipped.", "warning"))
        stats.skipped_records += 1
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        stats.invalid_records += 1
        stats.issues.append(DatasetIssue(source, "Invalid UTF-8 recovered with replacement characters."))
        try:
            return data.decode("utf-8", errors="replace")
        except Exception as exc:
            stats.issues.append(DatasetIssue(source, f"Could not decode text: {exc}", "error"))
            return None


def _iter_text_file(path: Path, stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        data = path.read_bytes()
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read file: {exc}", "error"))
        return
    text = _decode_bytes(data, str(path), stats)
    if text:
        yield DatasetRecord(normalize_text(text), str(path), {"format": "text", "suffix": path.suffix})


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
        if "instruction" in row and any(key in row for key in ("input", "output", "response")):
            prompt = "\n".join(_coerce_cell(part) for part in (row.get("instruction"), row.get("input")) if part)
            response = _coerce_cell(row.get("output", row.get("response")))
            return f"<|user|>\n{prompt}\n<|assistant|>\n{response}" if response else prompt
        if "prompt" in row and any(key in row for key in ("completion", "response")):
            return "\n".join(_coerce_cell(row[key]) for key in ("prompt", "completion", "response") if row.get(key))
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
    return "generative"


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
                role = msg.get("role", msg.get("from", "user"))
                if role == "human":
                    role = "user"
                if role == "gpt":
                    role = "assistant"
                content = msg.get("content", msg.get("value", ""))
                parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def _iter_jsonl(
    path: Path,
    text_key: str,
    text_columns: Iterable[str],
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats.issues.append(DatasetIssue(str(path), f"Bad JSONL line {line_no}: {exc}"))
                    stats.invalid_records += 1
                    continue
                if isinstance(row, dict):
                    stats.add_schema(str(path), row.keys())
                    for key in row:
                        stats.add_field(key)
                text = _extract_text_from_row(row, text_key, text_columns)
                if text:
                    yield DatasetRecord(
                        sanitize_text(text),
                        str(path),
                        {"format": "jsonl", "line": line_no, "mode": _infer_row_mode(row)},
                    )
                else:
                    stats.skipped_records += 1
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read JSONL: {exc}", "error"))


def _iter_json(
    path: Path,
    text_key: str,
    text_columns: Iterable[str],
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read JSON: {exc}", "error"))
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
                {"format": "json", "index": idx, "mode": _infer_row_mode(row)},
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
                    {"format": "csv", "row": idx, "mode": _infer_row_mode(row)},
                )
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read CSV/TSV: {exc}", "error"))


def _iter_yaml(path: Path, text_key: str, text_columns: Iterable[str], stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        import yaml
    except Exception:
        yield from _iter_text_file(path, stats)
        return
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read YAML: {exc}", "error"))
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
                {"format": "yaml", "index": idx, "mode": _infer_row_mode(row)},
            )


def _iter_xml(path: Path, stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        root = ElementTree.parse(path).getroot()
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read XML: {exc}", "error"))
        return
    for idx, node in enumerate(root.iter()):
        text = " ".join(t.strip() for t in node.itertext() if t.strip())
        if text:
            yield DatasetRecord(normalize_text(text), str(path), {"format": "xml", "index": idx, "tag": node.tag})


def _iter_sqlite(
    path: Path,
    text_key: str,
    text_columns: Iterable[str],
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        tables = con.execute("select name from sqlite_master where type='table'").fetchall()
        for (table,) in tables:
            cols = [r[1] for r in con.execute(f"pragma table_info({table})").fetchall()]
            stats.add_schema(f"{path}:{table}", cols)
            selected = _select_columns(cols, text_key, text_columns)
            if not selected:
                continue
            stats.add_text_columns(f"{path}:{table}", selected)
            quoted = ", ".join(f'"{col}"' for col in selected)
            for idx, values in enumerate(con.execute(f"select {quoted} from {table}")):
                text = "\n".join(_coerce_cell(value) for value in values if value)
                if text:
                    yield DatasetRecord(sanitize_text(text), str(path), {"format": "sqlite", "table": table, "row": idx})
                else:
                    stats.skipped_records += 1
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read SQLite: {exc}", "error"))


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
        import pyarrow.parquet as pq
        import pyarrow.ipc as ipc
    except Exception:
        stats.issues.append(DatasetIssue(str(path), f"Install pyarrow to read {fmt} datasets."))
        return
    try:
        if fmt == "parquet":
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
            selected = _select_columns(names, text_key, text_columns)
            stats.add_schema(str(path), names)
            stats.add_text_columns(str(path), selected)
            if not selected:
                stats.issues.append(DatasetIssue(str(path), "No usable text column found."))
                return
            for batch_id, batch in enumerate(parquet.iter_batches(batch_size=batch_size, columns=selected)):
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
                stats.issues.append(DatasetIssue(str(path), "No usable text column found."))
                return
            for batch_id, batch in enumerate(batches):
                yield from _records_from_arrow_batch(
                    batch.select(selected), str(path), fmt, batch_id, text_key, text_columns, stats
                )
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read {fmt}: {exc}", "error"))


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
                {"format": fmt, "batch": batch_id, "row": row_id, "mode": _infer_row_mode(row)},
            )
        else:
            stats.skipped_records += 1


def _iter_archive(
    path: Path,
    text_key: str,
    text_columns: Iterable[str],
    code_only: bool,
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                for name in zf.namelist():
                    suffix = Path(name).suffix.lower()
                    if code_only and suffix not in CODE_EXTENSIONS:
                        continue
                    if suffix not in TEXT_EXTENSIONS and suffix not in CODE_EXTENSIONS and suffix not in STRUCTURED_EXTENSIONS:
                        continue
                    data = zf.read(name)
                    yield from _records_from_archive_bytes(data, f"{path}!{name}", text_key, text_columns, stats)
        else:
            with tarfile.open(path) as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    suffix = Path(member.name).suffix.lower()
                    if code_only and suffix not in CODE_EXTENSIONS:
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    yield from _records_from_archive_bytes(fh.read(), f"{path}!{member.name}", text_key, text_columns, stats)
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read archive: {exc}", "error"))


def _records_from_archive_bytes(
    data: bytes,
    source: str,
    text_key: str,
    text_columns: Iterable[str],
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    text = _decode_bytes(data, source, stats)
    if not text:
        return
    suffix = Path(source).suffix.lower()
    if suffix == ".jsonl":
        for line_no, line in enumerate(io.StringIO(text), 1):
            try:
                row = json.loads(line)
            except Exception:
                stats.invalid_records += 1
                continue
            value = _extract_text_from_row(row, text_key, text_columns)
            if value:
                yield DatasetRecord(
                    sanitize_text(value),
                    source,
                    {"format": "archive-jsonl", "line": line_no, "mode": _infer_row_mode(row)},
                )
            else:
                stats.skipped_records += 1
    else:
        yield DatasetRecord(sanitize_text(text), source, {"format": "archive-text"})


def _iter_http(url: str, text_key: str, text_columns: Iterable[str], stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        req = Request(url, headers={"User-Agent": "nanoforge/0.1"})
        with urlopen(req, timeout=30) as response:
            data = response.read()
    except Exception as exc:
        stats.issues.append(DatasetIssue(url, f"Could not read HTTP dataset: {exc}", "error"))
        return
    yield from _records_from_archive_bytes(data, url, text_key, text_columns, stats)


def _iter_huggingface(
    ref: str,
    text_key: str,
    text_columns: Iterable[str],
    stats: DatasetStats,
) -> Iterator[DatasetRecord]:
    try:
        from datasets import load_dataset
    except Exception:
        stats.issues.append(DatasetIssue(ref, "Install datasets to stream Hugging Face datasets."))
        return
    name = ref.removeprefix("hf://").removeprefix("hf:")
    if not name:
        stats.issues.append(DatasetIssue(ref, "Hugging Face dataset reference is empty.", "error"))
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
                    {"format": "huggingface", "row": idx, "mode": _infer_row_mode(row)},
                )
            else:
                stats.skipped_records += 1
    except Exception as exc:
        stats.issues.append(DatasetIssue(ref, f"Could not stream Hugging Face dataset: {exc}", "error"))
