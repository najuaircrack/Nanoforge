from __future__ import annotations

import ast
from pathlib import Path


def test_python_sources_parse() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root / "src").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

