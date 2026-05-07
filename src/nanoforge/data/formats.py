from __future__ import annotations

import csv
import io
import json
import sqlite3
import tarfile
import zipfile
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
    issues: list[DatasetIssue] = field(default_factory=list)
    fields: dict[str, int] = field(default_factory=dict)
    formats: dict[str, int] = field(default_factory=dict)

    def add_field(self, name: str) -> None:
        self.fields[name] = self.fields.get(name, 0) + 1

    def add_format(self, name: str) -> None:
        self.formats[name] = self.formats.get(name, 0) + 1


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
    code_only: bool = False,
    stats: DatasetStats | None = None,
) -> Iterator[DatasetRecord]:
    stats = stats or DatasetStats()
    for raw in paths:
        fmt = detect_format(raw)
        stats.add_format(fmt)
        if fmt == "directory":
            yield from _iter_directory(Path(raw), text_key=text_key, code_only=code_only, stats=stats)
        elif fmt == "archive":
            yield from _iter_archive(Path(raw), text_key=text_key, code_only=code_only, stats=stats)
        elif fmt == "jsonl":
            yield from _iter_jsonl(Path(raw), text_key=text_key, stats=stats)
        elif fmt == "json":
            yield from _iter_json(Path(raw), text_key=text_key, stats=stats)
        elif fmt == "csv":
            yield from _iter_csv(Path(raw), text_key=text_key, stats=stats)
        elif fmt == "yaml":
            yield from _iter_yaml(Path(raw), text_key=text_key, stats=stats)
        elif fmt == "xml":
            yield from _iter_xml(Path(raw), stats=stats)
        elif fmt == "sqlite":
            yield from _iter_sqlite(Path(raw), text_key=text_key, stats=stats)
        elif fmt in {"parquet", "arrow"}:
            yield from _iter_arrow_like(Path(raw), fmt=fmt, text_key=text_key, stats=stats)
        elif fmt == "huggingface":
            yield from _iter_huggingface(str(raw), text_key=text_key, stats=stats)
        elif fmt.startswith("http"):
            yield from _iter_http(str(raw), text_key=text_key, stats=stats)
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


def _iter_directory(path: Path, text_key: str, code_only: bool, stats: DatasetStats) -> Iterator[DatasetRecord]:
    for file in path.rglob("*"):
        if file.is_file():
            if code_only and file.suffix.lower() not in CODE_EXTENSIONS:
                continue
            yield from iter_dataset_records([file], text_key=text_key, code_only=code_only, stats=stats)


def _decode_bytes(data: bytes, source: str, stats: DatasetStats) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
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


def _extract_text_from_row(row: Any, text_key: str) -> str | None:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        if text_key in row:
            return str(row[text_key])
        if "messages" in row:
            return _format_messages(row["messages"])
        for key in ("content", "prompt", "completion", "instruction", "response", "code"):
            if key in row:
                return str(row[key])
    return None


def _format_messages(messages: Any) -> str:
    parts = []
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                parts.append(f"<|{msg.get('role', 'user')}|>\n{msg.get('content', '')}")
    return "\n".join(parts)


def _iter_jsonl(path: Path, text_key: str, stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    stats.issues.append(DatasetIssue(str(path), f"Bad JSONL line {line_no}: {exc}"))
                    continue
                text = _extract_text_from_row(row, text_key)
                if text:
                    yield DatasetRecord(normalize_text(text), str(path), {"format": "jsonl", "line": line_no})
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read JSONL: {exc}", "error"))


def _iter_json(path: Path, text_key: str, stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read JSON: {exc}", "error"))
        return
    rows = data if isinstance(data, list) else data.get("data", [data]) if isinstance(data, dict) else [data]
    for idx, row in enumerate(rows):
        text = _extract_text_from_row(row, text_key)
        if text:
            yield DatasetRecord(normalize_text(text), str(path), {"format": "json", "index": idx})


def _iter_csv(path: Path, text_key: str, stats: DatasetStats) -> Iterator[DatasetRecord]:
    dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh, dialect=dialect)
            for idx, row in enumerate(reader):
                text = _extract_text_from_row(row, text_key)
                if text is None:
                    text = "\n".join(str(v) for v in row.values() if v)
                yield DatasetRecord(normalize_text(text), str(path), {"format": "csv", "row": idx})
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read CSV/TSV: {exc}", "error"))


def _iter_yaml(path: Path, text_key: str, stats: DatasetStats) -> Iterator[DatasetRecord]:
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
        text = _extract_text_from_row(row, text_key)
        if text:
            yield DatasetRecord(normalize_text(text), str(path), {"format": "yaml", "index": idx})


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


def _iter_sqlite(path: Path, text_key: str, stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        tables = con.execute("select name from sqlite_master where type='table'").fetchall()
        for (table,) in tables:
            cols = [r[1] for r in con.execute(f"pragma table_info({table})").fetchall()]
            candidate = text_key if text_key in cols else next((c for c in cols if c in {"text", "content", "body"}), None)
            if candidate is None:
                continue
            for idx, (text,) in enumerate(con.execute(f"select {candidate} from {table}")):
                if text:
                    yield DatasetRecord(normalize_text(str(text)), str(path), {"format": "sqlite", "table": table, "row": idx})
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read SQLite: {exc}", "error"))


def _iter_arrow_like(path: Path, fmt: str, text_key: str, stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        import pyarrow.parquet as pq
        import pyarrow.ipc as ipc
    except Exception:
        stats.issues.append(DatasetIssue(str(path), f"Install pyarrow to read {fmt} datasets."))
        return
    try:
        table = pq.read_table(path) if fmt == "parquet" else ipc.open_file(path).read_all()
        names = table.column_names
        column = text_key if text_key in names else next((c for c in names if c in {"text", "content", "body"}), None)
        if column is None:
            stats.issues.append(DatasetIssue(str(path), "No usable text column found."))
            return
        for idx, value in enumerate(table[column].to_pylist()):
            if value:
                yield DatasetRecord(normalize_text(str(value)), str(path), {"format": fmt, "row": idx})
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read {fmt}: {exc}", "error"))


def _iter_archive(path: Path, text_key: str, code_only: bool, stats: DatasetStats) -> Iterator[DatasetRecord]:
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
                    yield from _records_from_archive_bytes(data, f"{path}!{name}", text_key, stats)
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
                    yield from _records_from_archive_bytes(fh.read(), f"{path}!{member.name}", text_key, stats)
    except Exception as exc:
        stats.issues.append(DatasetIssue(str(path), f"Could not read archive: {exc}", "error"))


def _records_from_archive_bytes(data: bytes, source: str, text_key: str, stats: DatasetStats) -> Iterator[DatasetRecord]:
    text = _decode_bytes(data, source, stats)
    if not text:
        return
    suffix = Path(source).suffix.lower()
    if suffix == ".jsonl":
        for line_no, line in enumerate(io.StringIO(text), 1):
            try:
                row = json.loads(line)
            except Exception:
                continue
            value = _extract_text_from_row(row, text_key)
            if value:
                yield DatasetRecord(normalize_text(value), source, {"format": "archive-jsonl", "line": line_no})
    else:
        yield DatasetRecord(normalize_text(text), source, {"format": "archive-text"})


def _iter_http(url: str, text_key: str, stats: DatasetStats) -> Iterator[DatasetRecord]:
    try:
        req = Request(url, headers={"User-Agent": "nanoforge/0.1"})
        with urlopen(req, timeout=30) as response:
            data = response.read()
    except Exception as exc:
        stats.issues.append(DatasetIssue(url, f"Could not read HTTP dataset: {exc}", "error"))
        return
    yield from _records_from_archive_bytes(data, url, text_key, stats)


def _iter_huggingface(ref: str, text_key: str, stats: DatasetStats) -> Iterator[DatasetRecord]:
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
            text = _extract_text_from_row(row, text_key)
            if text:
                yield DatasetRecord(normalize_text(text), ref, {"format": "huggingface", "row": idx})
    except Exception as exc:
        stats.issues.append(DatasetIssue(ref, f"Could not stream Hugging Face dataset: {exc}", "error"))
