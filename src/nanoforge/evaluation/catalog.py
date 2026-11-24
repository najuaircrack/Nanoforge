"""Bounded evaluation catalogs and comparison thresholds."""
from __future__ import annotations
import json
import os
from pathlib import Path
from nanoforge.artifacts import fingerprint_file
from nanoforge.evaluation.suite import load_task_suite

def _files(root,suffix,max_files):
    root = Path(root).resolve()
    if max_files < 1:
        raise ValueError("max_files must be positive")
    count = 0
    scanned = 0
    for directory,dirs,names in os.walk(root,followlinks=False):
        scanned += 1 + len(names) + len(dirs)
        if scanned > max_files * 100:
            raise ValueError("catalog scan budget exceeded; choose a narrower root")
        dirs[:] = sorted(d for d in dirs if d not in {".git",".venv","node_modules",".unlazy",".evaluations",".generations","models","data"})
        for name in sorted(names):
            path = Path(directory)/name
            if not name.endswith(suffix) or path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            if path.stat().st_size > 33554432:
                continue
            yield root,path
            count += 1
            if count >= max_files:
                return

def list_task_suites(root,*,max_files=1000):
    result = []
    for base,path in _files(root,".jsonl",max_files):
        try:
            suite = load_task_suite(path)
        except ValueError:
            continue
        result.append({"path":path.relative_to(base).as_posix(),"name":suite.suite_id,"count":len(suite.cases),"fingerprint":suite.fingerprint})
    return result

def load_evaluation_report(path):
    path = Path(path).resolve()
    if path.stat().st_size > 33554432:
        raise ValueError("report exceeds byte limit")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("report_type") not in {"task_suite","agent_suite"}:
        raise ValueError("not an evaluation report")
    version = report.get("schema_version",1)
    if isinstance(version,bool) or not isinstance(version,int) or not 1 <= version <= 2:
        raise ValueError("unsupported report schema")
    cases = report.get("cases")
    if not isinstance(cases,list) or not isinstance(report.get("summary"),dict):
        raise ValueError("invalid evaluation report")
    for entry in report.get("files",{}).values():
        relative = Path(entry["path"])
        sidecar = (path.parent/relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not sidecar.is_relative_to(path.parent) or not sidecar.is_file() or fingerprint_file(sidecar) != entry["fingerprint"]:
            raise ValueError("evaluation integrity failure")
    return {"report":report,"cases":cases,"failures":[c for c in cases if c.get("status") in {"failed","error"}]}

def list_evaluation_reports(root,*,max_files=1000):
    result = []
    for base,path in _files(root,".json",max_files):
        try:
            report = load_evaluation_report(path)["report"]
        except (ValueError,KeyError,TypeError):
            continue
        result.append({"path":path.relative_to(base).as_posix(),"summary":report["summary"],"suite_id":report["suite_id"],"suite_fingerprint":report["suite_fingerprint"],"report_id":report["report_id"]})
    return result

def compare_evaluation_reports(paths,*,thresholds=None):
    paths = list(paths)
    if not 2 <= len(paths) <= 32:
        raise ValueError("comparison requires 2-32 reports")
    defaults = {"min_pass_rate_delta":0.0,"max_regressions":0,"max_mean_latency_delta_ms":None,"max_peak_rss_mb":None}
    if set(thresholds or {})-defaults.keys():
        raise ValueError("unknown comparison threshold")
    thresholds = {**defaults,**(thresholds or {})}
    reports = [load_evaluation_report(p)["report"] for p in paths]
    baseline = reports[0]
    before = {c["case_id"]:c for c in baseline["cases"]}
    candidates = []
    for report in reports[1:]:
        after = {c["case_id"]:c for c in report["cases"]}
        if report["suite_fingerprint"] != baseline["suite_fingerprint"] or before.keys() != after.keys():
            raise ValueError("reports use different suite definitions")
        regressions = [k for k in before if before[k]["status"] == "passed" and after[k]["status"] != "passed"]
        improvements = [k for k in before if before[k]["status"] != "passed" and after[k]["status"] == "passed"]
        delta = report["summary"]["pass_rate"]-baseline["summary"]["pass_rate"]
        latency = sum(c["latency_ms"] for c in after.values())/len(after)-sum(c["latency_ms"] for c in before.values())/len(before)
        passed = delta >= thresholds["min_pass_rate_delta"] and len(regressions) <= thresholds["max_regressions"]
        if thresholds["max_mean_latency_delta_ms"] is not None:
            passed = passed and latency <= thresholds["max_mean_latency_delta_ms"]
        peak = report.get("resources",{}).get("run_peak_sampled_rss_mb")
        if thresholds["max_peak_rss_mb"] is not None:
            passed = passed and peak is not None and peak <= thresholds["max_peak_rss_mb"]
        candidates.append({"report_id":report["report_id"],"pass_rate_delta":delta,"mean_latency_delta_ms":latency,"regressions":regressions,"improvements":improvements,"thresholds_passed":bool(passed)})
    return {"compatible":True,"baseline":baseline["report_id"],"candidates":candidates,"thresholds":thresholds}
