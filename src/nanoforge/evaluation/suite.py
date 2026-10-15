"""Deterministic JSONL task-suite evaluation.

The suite intentionally checks observable contracts rather than pretending to
measure broad intelligence. Tool-call checks validate JSON only; they never
execute a tool. Grounding checks are lexical/citation checks, not semantic
faithfulness judgments.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nanoforge.artifacts import (
    atomic_write_json,
    canonical_json,
    content_fingerprint,
    environment_snapshot,
    resource_snapshot,
)
from nanoforge.generation.sampling import SamplingConfig
from nanoforge.training.utils import seed_everything

SUITE_SCHEMA_VERSION = 1


class TextGenerator(Protocol):
    def complete(
        self,
        prompt: str,
        max_new_tokens: int,
        sampling: SamplingConfig,
        stop_tokens: list[str] | None = None,
    ) -> str: ...


@dataclass
class TaskCase:
    case_id: str
    task_type: str
    prompt: str
    expected: Any = None
    context: Any = None
    generation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSuite:
    suite_id: str
    cases: list[TaskCase]
    schema_version: int = SUITE_SCHEMA_VERSION
    fingerprint: str = ""


@dataclass
class CheckResult:
    passed: bool
    score: float
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    task_type: str
    status: str
    output: str
    check: CheckResult
    latency_ms: float
    error: str | None = None


@dataclass
class EvaluationReport:
    schema_version: int
    report_type: str
    report_id: str
    suite_id: str
    suite_fingerprint: str
    model: dict[str, Any]
    environment: dict[str, Any]
    configuration: dict[str, Any]
    summary: dict[str, Any]
    cases: list[CaseResult]
    resources: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _case_from_dict(raw: dict[str, Any], line_number: int) -> TaskCase:
    if not isinstance(raw, dict):
        raise ValueError(f"Suite line {line_number}: expected a JSON object.")
    missing = [key for key in ("id", "type", "prompt") if key not in raw]
    if missing:
        raise ValueError(f"Suite line {line_number}: missing required field(s): {', '.join(missing)}")
    if not isinstance(raw["id"], str) or not raw["id"].strip():
        raise ValueError(f"Suite line {line_number}: id must be a non-empty string.")
    if not isinstance(raw["type"], str) or not raw["type"].strip():
        raise ValueError(f"Suite line {line_number}: type must be a non-empty string.")
    if not isinstance(raw["prompt"], str):
        raise ValueError(f"Suite line {line_number}: prompt must be a string.")
    generation = raw.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError(f"Suite line {line_number}: generation must be an object.")
    metadata = dict(raw.get("metadata", {}))
    if "tags" in raw:
        metadata["tags"] = raw["tags"]
    return TaskCase(
        case_id=raw["id"],
        task_type=raw["type"].strip().lower().replace("-", "_"),
        prompt=raw["prompt"],
        expected=raw.get("expected"),
        context=raw.get("context"),
        generation=generation,
        metadata=metadata,
    )


def load_task_suite(path: str | Path) -> TaskSuite:
    path = Path(path)
    cases: list[TaskCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Suite line {line_number}: invalid JSON: {exc.msg}") from exc
            case = _case_from_dict(raw, line_number)
            if case.case_id in seen:
                raise ValueError(f"Suite line {line_number}: duplicate case id {case.case_id!r}.")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"Suite {path} contains no cases.")
    normalized = [asdict(case) for case in cases]
    return TaskSuite(
        suite_id=path.stem,
        cases=cases,
        fingerprint=content_fingerprint(normalized),
    )


def _normalized_text(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def check_exact_match(output: str, expected: str, *, normalize: bool = False) -> CheckResult:
    actual = _normalized_text(output) if normalize else output.strip()
    target = _normalized_text(expected) if normalize else expected.strip()
    passed = actual == target
    return CheckResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="exact match" if passed else "output did not match expected text",
        details={"expected": expected, "actual": output, "normalized": normalize},
    )


def check_structured_output(output: str, expected: Any) -> CheckResult:
    try:
        actual = json.loads(output)
    except json.JSONDecodeError as exc:
        return CheckResult(False, 0.0, "output is not valid JSON", {"error": exc.msg, "output": output})
    passed = actual == expected
    return CheckResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="structured output matched" if passed else "structured output differed",
        details={"expected": expected, "actual": actual},
    )


def check_tool_call(output: str, expected: dict[str, Any]) -> CheckResult:
    try:
        actual = json.loads(output)
    except json.JSONDecodeError as exc:
        return CheckResult(False, 0.0, "tool call is not valid JSON", {"error": exc.msg, "output": output})
    if not isinstance(actual, dict):
        return CheckResult(False, 0.0, "tool call must be a JSON object", {"actual": actual})
    actual_name = actual.get("name")
    actual_arguments = actual.get("arguments", {})
    expected_arguments = expected.get("arguments", {})
    passed = actual_name == expected.get("name") and actual_arguments == expected_arguments
    return CheckResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="tool call matched" if passed else "tool name or arguments differed",
        details={"expected": expected, "actual": actual, "executed": False},
    )


def check_grounding(output: str, expected: dict[str, Any]) -> CheckResult:
    required_citations = [str(item) for item in expected.get("required_citations", [])]
    required_substrings = [str(item) for item in expected.get("required_substrings", [])]
    missing_citations = [item for item in required_citations if item not in output]
    missing_substrings = [item for item in required_substrings if item.casefold() not in output.casefold()]
    total = len(required_citations) + len(required_substrings)
    passed_count = total - len(missing_citations) - len(missing_substrings)
    passed = not missing_citations and not missing_substrings
    return CheckResult(
        passed=passed,
        score=1.0 if total == 0 else passed_count / total,
        reason="grounding requirements matched" if passed else "grounding requirements missing",
        details={
            "missing_citations": missing_citations,
            "missing_substrings": missing_substrings,
            "semantic_faithfulness_checked": False,
        },
    )


def _sampling_config(options: dict[str, Any]) -> SamplingConfig:
    fields = set(SamplingConfig.__dataclass_fields__)
    values = {key: value for key, value in options.items() if key in fields}
    return SamplingConfig(**values)


def _check_case(case: TaskCase, output: str) -> CheckResult:
    if case.task_type in {"exact_match", "exact"}:
        return check_exact_match(output, str(case.expected), normalize=bool(case.generation.get("normalize", False)))
    if case.task_type in {"structured", "json"}:
        return check_structured_output(output, case.expected)
    if case.task_type in {"tool_call", "tool"}:
        if not isinstance(case.expected, dict):
            return CheckResult(False, 0.0, "tool_call expected value must be an object")
        return check_tool_call(output, case.expected)
    if case.task_type in {"grounding", "grounded"}:
        if not isinstance(case.expected, dict):
            return CheckResult(False, 0.0, "grounding expected value must be an object")
        return check_grounding(output, case.expected)
    return CheckResult(False, 0.0, f"unsupported task type: {case.task_type}")


def evaluate_task_suite(
    suite: TaskSuite,
    generator: TextGenerator,
    *,
    model_metadata: dict[str, Any] | None = None,
    device: str = "auto",
    seed: int = 1337,
) -> EvaluationReport:
    seed_everything(seed)
    results: list[CaseResult] = []
    for case in suite.cases:
        options = dict(case.generation)
        max_new_tokens = int(options.pop("max_new_tokens", 128))
        stop_tokens = options.pop("stop_tokens", None)
        started = time.perf_counter()
        output = ""
        try:
            output = generator.complete(
                case.prompt,
                max_new_tokens=max_new_tokens,
                sampling=_sampling_config(options),
                stop_tokens=stop_tokens,
            )
            check = _check_case(case, output)
            status = "passed" if check.passed else "failed"
            error = None
        except Exception as exc:  # A failed case must remain visible in the report.
            check = CheckResult(False, 0.0, "generator error", {})
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            CaseResult(
                case_id=case.case_id,
                task_type=case.task_type,
                status=status,
                output=output,
                check=check,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=error,
            )
        )
    passed = sum(result.status == "passed" for result in results)
    failed = sum(result.status == "failed" for result in results)
    errors = sum(result.status == "error" for result in results)
    summary = {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": passed / len(results),
        "seed": seed,
    }
    report = EvaluationReport(
        schema_version=SUITE_SCHEMA_VERSION,
        report_type="task_suite",
        report_id="",
        suite_id=suite.suite_id,
        suite_fingerprint=suite.fingerprint,
        model=model_metadata or {},
        environment=environment_snapshot(device),
        configuration={"seed": seed},
        summary=summary,
        cases=results,
        resources=resource_snapshot(),
    )
    report.report_id = content_fingerprint({key: value for key, value in report.to_dict().items() if key != "report_id"})
    return report


def write_evaluation_report(report: EvaluationReport, output: str | Path) -> dict[str, Path]:
    output = Path(output)
    report_dir = output if output.suffix == "" else output.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.json" if output.suffix == "" else output
    cases_path = report_path.with_name("cases.jsonl")
    failures_path = report_path.with_name("failures.jsonl")
    atomic_write_json(report_path, report.to_dict())
    case_lines = "".join(canonical_json(asdict(case)) + "\n" for case in report.cases)
    failure_lines = "".join(
        canonical_json(asdict(case)) + "\n" for case in report.cases if case.status in {"failed", "error"}
    )
    cases_path.write_text(case_lines, encoding="utf-8")
    failures_path.write_text(failure_lines, encoding="utf-8")
    return {"report": report_path, "cases": cases_path, "failures": failures_path}
