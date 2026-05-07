from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def reset_metric_file(path: str | Path, backup: bool = True) -> Path | None:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return None
    if not backup:
        path.write_text("", encoding="utf-8")
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.stem}.{stamp}.jsonl")
    shutil.move(str(path), str(backup_path))
    path.write_text("", encoding="utf-8")
    return backup_path


class JsonlMetricLogger:
    """Append-only metric stream used by CLI progress, dashboards, and recovery tools."""

    def __init__(self, path: str | Path, reset: bool = False, backup: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_path = reset_metric_file(self.path, backup=backup) if reset else None

    def log(self, event: str, step: int, metrics: dict[str, Any]) -> None:
        row = json_safe({"time": time.time(), "event": event, "step": step, **metrics})
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl_tail(path: str | Path, limit: int = 1000) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json_safe(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]
