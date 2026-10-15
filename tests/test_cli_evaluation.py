from __future__ import annotations

import json

from nanoforge import cli


class _Generator:
    def complete(self, prompt, max_new_tokens, sampling, stop_tokens=None):
        return "4"


def test_evaluate_suite_cli_writes_report(tmp_path, monkeypatch, capsys):
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps({"id": "one", "type": "exact_match", "prompt": "question", "expected": "4"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_load_generation_engine", lambda args: _Generator())
    cli.main(["evaluate-suite", "--suite", str(suite), "--model", "fake", "--out", str(tmp_path / "report")])
    output = capsys.readouterr().out
    assert "passed: 1/1" in output
    assert (tmp_path / "report" / "report.json").exists()
