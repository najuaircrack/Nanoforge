from __future__ import annotations

import json

import pytest

from nanoforge.evaluation.suite import (
    check_exact_match,
    check_grounding,
    check_structured_output,
    check_tool_call,
    evaluate_task_suite,
    load_task_suite,
    write_evaluation_report,
)


class FakeGenerator:
    def complete(self, prompt, max_new_tokens, sampling, stop_tokens=None):
        if prompt == "error":
            raise RuntimeError("synthetic failure")
        if prompt == "math":
            return "4"
        if prompt == "json":
            return '{"name": "Nanoforge"}'
        if prompt == "tool":
            return '{"name": "weather", "arguments": {"city": "London"}}'
        if prompt == "ground":
            return "Nanoforge is local-first. [doc-1]"
        return "wrong"


def test_checkers_cover_success_and_failure():
    assert check_exact_match("4", "4").passed
    assert not check_exact_match("5", "4").passed
    assert check_structured_output('{"a": 1}', {"a": 1}).passed
    assert not check_structured_output("not json", {"a": 1}).passed
    assert check_tool_call('{"name":"x","arguments":{}}', {"name": "x", "arguments": {}}).passed
    assert not check_tool_call('{"name":"x","arguments":{}}', {"name": "y", "arguments": {}}).passed
    assert check_grounding("answer [doc-1] local-first", {"required_citations": ["doc-1"], "required_substrings": ["local-first"]}).passed
    assert not check_grounding("answer", {"required_citations": ["doc-1"]}).passed


def test_suite_report_preserves_failures_and_errors(tmp_path):
    suite_path = tmp_path / "smoke.jsonl"
    cases = [
        {"id": "math", "type": "exact_match", "prompt": "math", "expected": "4", "generation": {"mode": "deterministic"}},
        {"id": "json", "type": "structured", "prompt": "json", "expected": {"name": "Nanoforge"}},
        {"id": "tool", "type": "tool_call", "prompt": "tool", "expected": {"name": "weather", "arguments": {"city": "London"}}},
        {"id": "ground", "type": "grounding", "prompt": "ground", "expected": {"required_citations": ["doc-1"], "required_substrings": ["local-first"]}},
        {"id": "failed", "type": "exact_match", "prompt": "wrong", "expected": "expected"},
        {"id": "error", "type": "exact_match", "prompt": "error", "expected": "never"},
    ]
    suite_path.write_text("\n".join(json.dumps(case) for case in cases) + "\n", encoding="utf-8")
    suite = load_task_suite(suite_path)
    report = evaluate_task_suite(suite, FakeGenerator(), device="cpu", seed=7)
    paths = write_evaluation_report(report, tmp_path / "report")

    assert report.summary == {"total": 6, "passed": 4, "failed": 1, "errors": 1, "pass_rate": 4 / 6, "seed": 7}
    assert paths["report"].exists()
    assert paths["cases"].read_text(encoding="utf-8").count("\n") == 6
    assert paths["failures"].read_text(encoding="utf-8").count("\n") == 2
    assert all(case.latency_ms >= 0 for case in report.cases)


def test_malformed_suite_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"ok","type":"exact_match","prompt":"x","expected":"x"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Suite line 2"):
        load_task_suite(path)
