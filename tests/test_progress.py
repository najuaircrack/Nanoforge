from __future__ import annotations

import json
import math

from nanoforge.progress import JsonlMetricLogger, read_jsonl_tail


def test_metric_logger_sanitizes_non_finite_values(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = JsonlMetricLogger(path)
    logger.log("train", 1, {"loss": math.nan, "nested": {"value": math.inf}})

    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    row = json.loads(raw)
    assert row["loss"] is None
    assert row["nested"]["value"] is None


def test_metric_reader_sanitizes_old_nan_rows(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1, "val/best_loss": NaN}\n', encoding="utf-8")
    rows = read_jsonl_tail(path)
    assert rows[0]["val/best_loss"] is None


def test_metric_logger_reset_backs_up_old_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 99}\n', encoding="utf-8")
    logger = JsonlMetricLogger(path, reset=True)
    logger.log("start", 0, {"loss": 1.0})

    assert json.loads(path.read_text(encoding="utf-8"))["step"] == 0
    backups = list(tmp_path.glob("metrics.*.jsonl"))
    assert len(backups) == 1
    assert '"step": 99' in backups[0].read_text(encoding="utf-8")
