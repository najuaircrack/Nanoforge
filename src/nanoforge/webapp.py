"""
Nanoforge Web App — full-scope FastAPI backend.
Serves the React SPA and provides all API endpoints.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from nanoforge.progress import json_safe, read_jsonl_tail
from nanoforge.serving import (
    EngineRunner,
    JobManager,
    SecurityMiddleware,
    ServingPolicy,
    WorkspacePaths,
    atomic_text,
)

try:
    from fastapi import UploadFile
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pydantic ships with fastapi; create_app raises a clean error if missing
    BaseModel = object  # type: ignore[assignment,misc]
    UploadFile = object
    ConfigDict = dict
    def Field(default=None, **kwargs):
        return default


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Request models (module level so FastAPI resolves them as request bodies) ──
class JobStartRequest(StrictRequest):
    type: Literal["train", "pretrain", "pipeline", "auto-train", "prepare", "download"]
    config: str | None = None
    input: list[str] | None = None
    out: str | None = None
    mode: str = "auto"
    tokenizer: str = "byte"
    seq_len: int = Field(512, ge=2, le=8192)
    dataset: str | None = None
    name: str | None = Field(None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    max_steps: int = Field(50000, ge=1, le=10000000)


class JobStopRequest(StrictRequest):
    job_id: str


class LoadRequest(StrictRequest):
    path: str


class GenerateRequest(StrictRequest):
    prompt: str = Field(max_length=1024 * 1024)
    temperature: float = Field(0.8, ge=0, le=2)
    top_k: int = Field(50, ge=0)
    max_new_tokens: int = Field(256, ge=1, le=4096)
    mode: Literal["balanced", "chat", "creative", "coding", "deterministic", "low_memory", "high_quality"] = "balanced"


class PathRequest(StrictRequest):
    path: str


class SaveConfigRequest(StrictRequest):
    path: str
    yaml_text: str


class ImportRequest(StrictRequest):
    source: str
    name: str
    backend: str | None = None
    tokenizer: str | None = None


class ChatStreamRequest(StrictRequest):
    messages: list[tuple[Literal["system", "user", "assistant", "tool"], Annotated[str, Field(max_length=1024 * 1024)]]] = Field(min_length=1, max_length=256)
    max_new_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0, le=2)
    top_k: int = Field(50, ge=0)


class RagLoadRequest(StrictRequest):
    path: str
    top_k: int = Field(3, ge=1, le=32)


class FactsLoadRequest(StrictRequest):
    path: str
    threshold: float = Field(0.5, ge=0, le=1)


class ConfigBuildRequest(StrictRequest):
    sections: dict[str, dict[str, Any]]  # {"model": {...}, "training": {...}, ...}
    path: str | None = None  # if set, also write the built YAML here


class ProjectSaveRequest(StrictRequest):
    path: str
    project: dict[str, Any]


class AgentStartRequest(StrictRequest):
    project_path: str
    request: str = Field(min_length=1, max_length=64 * 1024)
    actions: list[dict[str, Any]] | None = Field(None, max_length=64)


class ApprovalRequest(StrictRequest):
    approval_id: str
    approved: bool = Field(strict=True)


class ReplayRequest(StrictRequest):
    run_id: str | None = None
    path: str | None = None


class EvaluationRequest(StrictRequest):
    suite_path: str
    model_ref: str
    output_dir: str


class CompareRequest(StrictRequest):
    report_paths: list[str] = Field(min_length=2, max_length=16)


# ── Config schema introspection (drives the web param editor) ─────────────────
# Fields whose valid values come from a registry — the editor renders these as
# dropdowns populated live from ALL_REGISTRIES.
_REGISTRY_FIELDS: dict[tuple[str, str], str] = {
    ("model", "attention_backend"): "attention",
    ("model", "ffn_type"): "ffn",
    ("model", "normalization"): "normalization",
    ("model", "position_embedding"): "position",
    ("model", "block_type"): "block",
    ("model", "quantization_backend"): "quantization",
    ("model", "task_type"): "head",
    ("model", "activation"): "activation",
    ("training", "optimizer"): "optimizer",
    ("training", "scheduler"): "scheduler",
    ("data", "tokenizer_type"): "tokenizer",
    ("inference", "sampler"): "sampler",
}

# Non-registry enum fields with a fixed set of valid values.
_ENUM_FIELDS: dict[tuple[str, str], list[str]] = {
    ("training", "precision"): ["auto", "bf16", "fp16", "fp32"],
    ("training", "device"): ["auto", "cpu", "cuda"],
    ("training", "mode"): ["auto", "pretrain", "sft", "dpo", "distill"],
    ("training", "distributed_backend"): ["none", "ddp", "fsdp"],
    ("data", "mode"): ["generative", "chat", "instruct", "classification"],
    ("data", "loss_masking"): ["auto", "all", "assistant_only", "completion_only"],
    ("model", "input_type"): ["text", "tokens"],
    ("inference", "mode"): ["balanced", "chat", "creative", "coding", "deterministic", "low_memory", "high_quality"],
}

# Numeric slider/step hints for well-known bounded fields.
_NUMERIC_HINTS: dict[tuple[str, str], dict[str, float]] = {
    ("model", "dropout"): {"min": 0.0, "max": 1.0, "step": 0.01},
    ("model", "lora_dropout"): {"min": 0.0, "max": 1.0, "step": 0.01},
    ("training", "weight_decay"): {"min": 0.0, "max": 1.0, "step": 0.01},
    ("training", "warmup_ratio"): {"min": 0.0, "max": 1.0, "step": 0.01},
    ("training", "ema_decay"): {"min": 0.0, "max": 1.0, "step": 0.001},
    ("inference", "temperature"): {"min": 0.0, "max": 2.0, "step": 0.05},
    ("inference", "top_p"): {"min": 0.0, "max": 1.0, "step": 0.01},
    ("inference", "min_p"): {"min": 0.0, "max": 1.0, "step": 0.01},
    ("inference", "repetition_penalty"): {"min": 1.0, "max": 2.0, "step": 0.01},
}

_SECTION_LABELS: dict[str, str] = {
    "model": "Model", "training": "Training", "data": "Data", "inference": "Inference",
}


def _widget_for(ann: str, options: list[str] | None) -> str:
    if options:
        return "select"
    if "bool" in ann:
        return "toggle"
    if "int" in ann or "float" in ann:
        return "number"
    return "text"


def _config_schema() -> dict[str, Any]:
    """Introspect the config dataclasses into a UI-friendly field schema."""
    import dataclasses

    from nanoforge.config import DataConfig, InferenceConfig, ModelConfig, TrainConfig
    from nanoforge.registry import ALL_REGISTRIES, ensure_registry_ready

    ensure_registry_ready()
    section_types = {
        "model": ModelConfig, "training": TrainConfig,
        "data": DataConfig, "inference": InferenceConfig,
    }
    schema: dict[str, Any] = {}
    for section, cls in section_types.items():
        field_list: list[dict[str, Any]] = []
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                default = f.default_factory()  # type: ignore[misc]
            else:
                default = None
            if isinstance(default, tuple):
                default = list(default)
            ann = f.type if isinstance(f.type, str) else str(f.type)
            reg = _REGISTRY_FIELDS.get((section, f.name))
            options = ALL_REGISTRIES[reg].names() if reg else _ENUM_FIELDS.get((section, f.name))
            entry: dict[str, Any] = {
                "name": f.name,
                "type": ann,
                "default": json_safe(default),
                "widget": _widget_for(ann, options),
            }
            if options:
                entry["options"] = options
            if (section, f.name) in _NUMERIC_HINTS:
                entry["range"] = _NUMERIC_HINTS[(section, f.name)]
            field_list.append(entry)
        schema[section] = {"label": _SECTION_LABELS[section], "fields": field_list}
    return schema


def _run_summary(run_path: Path) -> dict[str, Any]:
    """Summarize a single run directory from its metrics tail + checkpoints."""
    metrics = run_path / "metrics.jsonl"
    latest: dict[str, Any] = {}
    step = None
    if metrics.exists():
        rows = read_jsonl_tail(metrics, limit=400)
        for row in rows:
            for k, v in row.items():
                if v is not None:
                    latest[k] = v
            if row.get("step") is not None:
                step = row["step"]
    ckpts = _scan_checkpoints(run_path)
    try:
        mtime = run_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return {
        "name": run_path.name,
        "path": str(run_path),
        "step": step,
        "train_loss": latest.get("train/loss"),
        "val_loss": latest.get("val/loss"),
        "checkpoints": len(ckpts),
        "has_metrics": metrics.exists(),
        "mtime": mtime,
    }


# ── RAG index holder (mirrors _engine_holder) ─────────────────────────────────
_rag_holder: dict = {"retriever": None, "path": None, "top_k": 3}
# Personal-facts answerer: answers known factual questions directly (retrieve +
# templated answer), bypassing the model. See rag/personal.py.
_facts_holder: dict = {"answerer": None, "path": None, "threshold": 0.5}
_engine_holder: dict = {"engine": None, "path": None}


# ── Dataset download registry (friendly id -> real HF Hub repo id) ────────────
_DATASET_REPOS: dict[str, str] = {
    "openwebtext": "Skylion007/openwebtext",
    "tinystories": "roneneldan/TinyStories",
    "the-pile": "monology/pile-uncopyrighted",
    "code-parrot": "codeparrot/codeparrot-clean",
    "alpaca": "tatsu-lab/alpaca",
    "dolly": "databricks/databricks-dolly-15k",
}
# Optional per-dataset config name passed as the 2nd positional arg to load_dataset.
_DATASET_CONFIGS: dict[str, str] = {}

# Friendly catalog metadata for the web download cards (id -> display info).
_DATASET_CATALOG: list[dict[str, str]] = [
    {"id": "tinystories", "name": "TinyStories", "size": "~1.5 GB",
     "desc": "Short synthetic children's stories — ideal for tiny LM smoke training."},
    {"id": "openwebtext", "name": "OpenWebText", "size": "~40 GB",
     "desc": "Open reproduction of WebText — general English web text."},
    {"id": "the-pile", "name": "The Pile (uncopyrighted)", "size": "~800 GB",
     "desc": "Large diverse corpus across 22 domains."},
    {"id": "code-parrot", "name": "CodeParrot (clean)", "size": "~50 GB",
     "desc": "Deduplicated Python source for code models."},
    {"id": "alpaca", "name": "Alpaca", "size": "~45 MB",
     "desc": "52k instruction-following pairs for SFT."},
    {"id": "dolly", "name": "Dolly 15k", "size": "~13 MB",
     "desc": "15k human instruction/response records for SFT."},
]

# ── System history ring buffers ───────────────────────────────────────────────
_CPU_HIST: collections.deque[float] = collections.deque(maxlen=180)
_RAM_HIST: collections.deque[float] = collections.deque(maxlen=180)

# ── Helpers (reused from dashboard) ──────────────────────────────────────────
def _scan_checkpoints(run_path: Path) -> list[dict]:
    ckpts = []
    for pt in sorted(run_path.glob("*.pt"), reverse=True):
        meta_path = pt.with_suffix(".pt.meta.json")
        entry: dict = {"name": pt.name, "path": str(pt), "step": None, "val_loss": None}
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                entry["step"] = m.get("step")
                entry["val_loss"] = m.get("val_loss")
            except Exception:
                pass
        ckpts.append(entry)
    return ckpts


def _flatten_config(cfg) -> dict:
    import dataclasses
    if cfg is None:
        return {}
    raw = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else (cfg if isinstance(cfg, dict) else {})
    result: dict = {}
    for section, vals in raw.items():
        if isinstance(vals, dict):
            result[section] = dict(vals)
        else:
            result.setdefault("misc", {})[section] = vals
    return result


def _load_engine_compat(path: str, *, paths: WorkspacePaths | None = None):
    try:
        from nanoforge.generation.engine import GenerationEngine
        if Path(path).suffix.lower() in {".pt", ".pth", ".ckpt"}:
            checkpoint = paths.resolve(path) if paths else Path(path)
            return GenerationEngine.from_checkpoint(checkpoint), None
        from nanoforge.imports import load_imported_engine, resolve_imported_model
        reference = path
        if paths:
            entry = resolve_imported_model(path, registry_file=paths.root / "models/imports.json")
            reference = entry.source
            local = paths.resolve(reference)
            if not local.exists():
                raise FileNotFoundError("Studio model import must resolve to an existing local model; network loading is disabled.")
            return load_imported_engine(path, registry_path=paths.root / "models/imports.json", allow_network=False), None
        return load_imported_engine(reference, allow_network=False), None
    except Exception as exc:
        return None, str(exc)


# ── App factory ───────────────────────────────────────────────────────────────
def create_app(run_dir: str = "runs/nanoforge", *, workspace_root: str | Path | None = None, policy: ServingPolicy | None = None):
    try:
        from fastapi import FastAPI, File, Form, HTTPException
        from fastapi.responses import HTMLResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except Exception as exc:
        raise RuntimeError("Install serve extras: pip install -e .[serve]") from exc

    paths = WorkspacePaths(workspace_root or Path.cwd())
    policy = policy or ServingPolicy()
    jobs = JobManager(paths.root, max_concurrency=policy.max_concurrency)
    runner = EngineRunner(policy)
    _engine_holder = dict(globals()["_engine_holder"])
    _rag_holder = dict(globals()["_rag_holder"])
    _facts_holder = dict(globals()["_facts_holder"])
    agent_runs: dict[str, dict[str, Any]] = {}
    agent_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        for run in list(agent_runs.values()):
            run["cancellation"].cancel()
        jobs.close()
        runner.close()

    app = FastAPI(title="Latticekin", lifespan=lifespan)
    app.add_middleware(SecurityMiddleware, policy=policy)
    app.state.jobs = jobs
    app.state.workspace = paths
    app.state.engine_holder = _engine_holder
    app.state.engine_runner = runner

    @app.exception_handler(ValueError)
    async def validation_error(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)

    @app.exception_handler(FileNotFoundError)
    async def missing_file(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)

    @app.exception_handler(PermissionError)
    async def forbidden_file(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=403)

    def confined(value, *, relative=False):
        try:
            return paths.resolve(value, allow_absolute=not relative)
        except (PermissionError, ValueError) as exc:
            raise HTTPException(403, str(exc)) from exc

    run_path = confined(run_dir)

    def metrics_for(job_id=None, run_dir=None):
        target = run_path
        if job_id:
            try:
                selected = jobs.status(job_id).get("run_dir")
            except KeyError as exc:
                raise HTTPException(404, "Job not found") from exc
            if selected:
                target = confined(selected)
        elif run_dir:
            target = confined(run_dir)
        return target, target / "metrics.jsonl"

    dist = Path(__file__).parent / "web_dist"

    # ── Metrics ───────────────────────────────────────────────────────────────
    @app.get("/api/metrics")
    def metrics(limit: int = 2000, job_id: str | None = None, run_dir: str | None = None):
        limit = max(1, min(limit, 10000))
        run_path, metrics_path = metrics_for(job_id, run_dir)
        rows = read_jsonl_tail(metrics_path, limit=limit)
        ckpts = _scan_checkpoints(run_path)
        cfg_data: dict = {}
        for src in [run_path / "config.yaml", *(Path(c["path"]).with_suffix(".yaml") for c in ckpts)]:
            if src.exists():
                try:
                    from nanoforge.config import load_config
                    cfg_data = _flatten_config(load_config(src))
                    break
                except Exception:
                    pass
        return json_safe({"run": run_path.name, "rows": rows, "checkpoints": ckpts, "config": cfg_data})

    @app.get("/api/metrics/summary")
    def metrics_summary(limit: int = 2000, job_id: str | None = None, run_dir: str | None = None):
        """Latest values + server-computed derived metrics (ETA, throughput, gaps)."""
        limit = max(1, min(limit, 10000))
        run_path, metrics_path = metrics_for(job_id, run_dir)
        rows = read_jsonl_tail(metrics_path, limit=limit)
        latest: dict[str, Any] = {}
        first_step = last_step = None
        first_time = last_time = None
        tps_samples: list[float] = []
        for row in rows:
            for k, v in row.items():
                if v is not None:
                    latest[k] = v
            step = row.get("step")
            if step is not None:
                if first_step is None:
                    first_step = step
                last_step = step
            ts = row.get("time") or row.get("timestamp")
            if ts is not None:
                if first_time is None:
                    first_time = ts
                last_time = ts
            tps = row.get("train/tokens_per_sec")
            if tps:
                tps_samples.append(float(tps))
        max_steps = latest.get("run/max_steps")
        elapsed = (last_time - first_time) if (first_time is not None and last_time is not None) else None
        progress = (last_step / max_steps) if (max_steps and last_step is not None) else None
        eta = None
        if elapsed and last_step and first_step is not None and max_steps:
            done = last_step - first_step
            if done > 0:
                eta = (elapsed / done) * (max_steps - last_step)
        mean_tps = sum(tps_samples) / len(tps_samples) if tps_samples else None
        best = latest.get("val/best_loss")
        val = latest.get("val/loss")
        summary = {
            "run": run_path.name,
            "step": last_step,
            "max_steps": max_steps,
            "progress": progress,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "train_loss": latest.get("train/loss"),
            "val_loss": val,
            "best_val_loss": best,
            "best_gap": (val - best) if (val is not None and best is not None) else None,
            "perplexity": latest.get("val/perplexity"),
            "lr": latest.get("train/lr"),
            "grad_norm": latest.get("train/grad_norm"),
            "tokens_per_sec": latest.get("train/tokens_per_sec"),
            "mean_tokens_per_sec": mean_tps,
            "tokens_seen": latest.get("train/tokens_seen"),
            "logit_entropy": latest.get("health/logit_entropy"),
            "parameters": latest.get("run/parameters"),
        }
        return json_safe({"ok": True, "summary": summary})

    @app.get("/api/runs")
    def list_runs(root: str = "runs"):
        """List run directories under ``root`` with a last-metric summary each."""
        base = confined(root)
        if not base.exists():
            return {"ok": True, "runs": []}
        runs = []
        for p in sorted(base.iterdir()):
            if p.is_dir():
                if len(runs) >= 500:
                    break
                if p.resolve().is_relative_to(paths.root):
                    runs.append(_run_summary(p))
        runs.sort(key=lambda r: r["mtime"], reverse=True)
        return json_safe({"ok": True, "runs": runs})

    # ── System ────────────────────────────────────────────────────────────────
    @app.get("/api/system")
    def system_stats():
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            vm = psutil.virtual_memory()
            _CPU_HIST.append(cpu)
            _RAM_HIST.append(vm.used / 1e9)
        except ImportError:
            return {"error": "psutil not installed — run: pip install psutil"}
        info: dict[str, Any] = {
            "cpu": cpu, "cpu_count": psutil.cpu_count(),
            "ram_used": vm.used, "ram_total": vm.total, "ram_percent": vm.percent,
            "cpu_history": list(_CPU_HIST), "ram_history": list(_RAM_HIST),
            "platform": sys.platform, "python": sys.version.split()[0],
        }
        try:
            import torch
            info["torch"] = torch.__version__
            info["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["cuda"] = torch.version.cuda
        except ImportError:
            pass
        return json_safe(info)

    # ── Job runner ────────────────────────────────────────────────────────────
    @app.post("/api/jobs/start")
    def job_start(req: JobStartRequest):
        py = sys.executable
        run_dir = None
        if req.type in ("train", "pretrain", "pipeline"):
            if not req.config:
                return {"ok": False, "error": "config required"}
            config_path = confined(req.config)
            if req.type != "pipeline":
                from nanoforge.config import load_config
                config = load_config(config_path, strict=True)
                run_dir = paths.relative(confined(config.training.out_dir))
                for value in (config.data.train_data, config.data.val_data, config.data.tokenizer_path):
                    if value:
                        confined(value)
            cmd = [py, "-m", "nanoforge", req.type, "--config", str(config_path)]
        elif req.type == "auto-train":
            if not req.input or not req.name:
                return {"ok": False, "error": "input and name required for auto-train"}
            input_paths = [str(confined(value)) for value in req.input]
            run_dir = "runs/" + req.name
            cmd = [py, "-m", "nanoforge", "auto-train", "--input", *input_paths, "--name", req.name,
                   "--mode", req.mode, "--tokenizer", req.tokenizer, "--seq-len", str(req.seq_len), "--max-steps", str(req.max_steps)]
        elif req.type == "prepare":
            if not req.input or not req.out:
                return {"ok": False, "error": "input and out required"}
            cmd = [py, "-m", "nanoforge", "prepare",
                   "--input", *(str(confined(value)) for value in req.input), "--out", str(confined(req.out)),
                   "--mode", req.mode, "--tokenizer", req.tokenizer,
                   "--seq-len", str(req.seq_len)]
        elif req.type == "download":
            if not req.dataset:
                return {"ok": False, "error": "dataset required"}
            repo = _DATASET_REPOS.get(req.dataset, req.dataset)
            from pathlib import PureWindowsPath
            if (paths.root / repo).exists() or Path(repo).is_absolute() or PureWindowsPath(repo).drive or ".." in Path(repo).parts:
                repo = str(confined(repo))
            import hashlib
            safe_name = hashlib.sha256(req.dataset.encode()).hexdigest()[:24]
            output = confined("data/hf/" + safe_name)
            cmd = [py, "-m", "nanoforge.serving", "download-dataset", "--source", repo, "--out", str(output)]
            if req.dataset in _DATASET_CONFIGS:
                cmd.extend(["--config", _DATASET_CONFIGS[req.dataset]])
        else:
            return {"ok": False, "error": f"unknown job type: {req.type}"}
        try:
            job_id = jobs.start_process(cmd, job_type=req.type, run_dir=run_dir)
        except RuntimeError as exc:
            raise HTTPException(429, str(exc)) from exc
        return {"ok": True, "job_id": job_id, "run_dir": run_dir}

    @app.get("/api/jobs")
    def job_list():
        return {"ok": True, "jobs": jobs.list()}

    @app.post("/api/jobs/stop")
    def job_stop(req: JobStopRequest):
        try:
            jobs.cancel(req.job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc
        return {"ok": True}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        try:
            return {"ok": True, **jobs.status(job_id)}
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    # ── SSE job stream ────────────────────────────────────────────────────────
    @app.get("/stream/job/{job_id}")
    def stream_job(job_id: str):
        def _gen():
            previous = []
            while True:
                try:
                    entry = jobs.status(job_id)
                except KeyError:
                    yield f"data: {json.dumps({'type':'done','code':-1})}\n\n"
                    return
                lines = entry["lines"]
                start = len(previous) if lines[:len(previous)] == previous else 0
                for line in lines[start:]:
                    yield f"data: {json.dumps({'type':'line','line':line})}\n\n"
                previous = lines
                if entry["done"]:
                    yield f"data: {json.dumps({'type':'done','code':entry['code']})}\n\n"
                    return
                time.sleep(0.2)
        return StreamingResponse(_gen(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── Chat / Inference ──────────────────────────────────────────────────────
    @app.get("/api/chat/status")
    def chat_status():
        return {"ok": True, "loaded": _engine_holder.get("path")}

    @app.post("/api/chat/load")
    def chat_load(req: LoadRequest):
        engine, err = _load_engine_compat(req.path, paths=paths)
        if err:
            return {"ok": False, "error": err}
        _engine_holder["engine"] = engine
        _engine_holder["path"] = req.path
        return {"ok": True}

    # ── RAG index ─────────────────────────────────────────────────────────────
    @app.get("/api/rag/status")
    def rag_status():
        return {"ok": True, "loaded": _rag_holder.get("path"), "top_k": _rag_holder.get("top_k")}

    @app.post("/api/rag/load")
    def rag_load(req: RagLoadRequest):
        try:
            from nanoforge.rag import load_index
            retriever = load_index(confined(req.path), engine=_engine_holder.get("engine"))
            _rag_holder["retriever"] = retriever
            _rag_holder["path"] = req.path
            _rag_holder["top_k"] = req.top_k
            return {"ok": True, "num_docs": retriever.store.meta.num_docs, "backend": retriever.store.meta.backend}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/rag/unload")
    def rag_unload():
        _rag_holder["retriever"] = None
        _rag_holder["path"] = None
        return {"ok": True}

    # ── Personal facts (retrieve + templated answer) ──────────────────────────
    @app.get("/api/facts/status")
    def facts_status():
        return {
            "ok": True,
            "loaded": _facts_holder.get("path"),
            "threshold": _facts_holder.get("threshold"),
        }

    @app.post("/api/facts/load")
    def facts_load(req: FactsLoadRequest):
        try:
            from nanoforge.rag.personal import PersonalAnswerer

            answerer = PersonalAnswerer.load(
                confined(req.path), engine=_engine_holder.get("engine"), threshold=req.threshold
            )
            if answerer is None:
                return {"ok": False, "error": f"could not load facts index at {req.path}"}
            _facts_holder["answerer"] = answerer
            _facts_holder["path"] = req.path
            _facts_holder["threshold"] = req.threshold
            return {"ok": True, "num_docs": answerer.retriever.store.meta.num_docs}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/facts/unload")
    def facts_unload():
        _facts_holder["answerer"] = None
        _facts_holder["path"] = None
        return {"ok": True}

    @app.post("/api/chat/stream")
    def chat_stream_sse(req: ChatStreamRequest):
        engine = _engine_holder.get("engine")
        if engine is None:
            def _err():
                yield f"data: {json.dumps({'type':'error','text':'No model loaded'})}\n\n"
            return StreamingResponse(_err(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        messages = [(r, c) for r, c in req.messages]

        # Personal-facts bypass: if a facts KB is loaded and the latest user turn
        # is a known factual question, answer it directly (retrieve + templated
        # answer) and skip the model entirely — the tiny chat model can't reliably
        # answer facts. Non-factual turns fall through to normal generation.
        answerer = _facts_holder.get("answerer")
        if answerer is not None:
            last_user = next((c for r, c in reversed(messages) if r == "user"), "")
            fact = answerer.answer(last_user) if last_user else None
            if fact is not None:
                def _fact_gen(text: str = fact):
                    yield f"data: {json.dumps({'type':'chunk','text':text,'source':'facts'})}\n\n"
                    yield f"data: {json.dumps({'type':'done'})}\n\n"
                return StreamingResponse(_fact_gen(), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # RAG injection — non-fatal; chat continues unchanged if retrieval fails
        system: str | None = None
        retrieval_warning = None
        retriever = _rag_holder.get("retriever")
        if retriever is not None:
            query = next((c for r, c in reversed(messages) if r == "user"), "")
            if query:
                try:
                    hits = retriever.query(query, top_k=_rag_holder.get("top_k", 3))
                    if hits:
                        blocks = "\n\n".join(f"[{i+1}] {h['text_chunk']}" for i, h in enumerate(hits))
                        system = f"Relevant context:\n{blocks}"
                except Exception as exc:
                    retrieval_warning = str(exc)

        def _gen():
            try:
                if retrieval_warning:
                    yield f"data: {json.dumps({'type':'warning','text':'Retrieval failed: ' + retrieval_warning})}\n\n"
                from nanoforge.generation.sampling import SamplingConfig
                sc = SamplingConfig(mode="chat", temperature=req.temperature, top_k=req.top_k)
                if hasattr(engine, "chat_stream"):
                    output = runner.stream(engine, messages, req.max_new_tokens, sc, system=system, stream_fn=engine.chat_stream)
                else:
                    from nanoforge.generation.engine import GenerationEngine
                    renderer = getattr(engine, "build_chat_prompt", GenerationEngine.build_chat_prompt)
                    output = runner.stream(engine, renderer(messages, system=system), req.max_new_tokens, sc)
                for chunk in output:
                    yield f"data: {json.dumps({'type':'chunk','text':chunk})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type':'error','text':str(exc)})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        return StreamingResponse(_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/chat/generate")
    def chat_generate(req: GenerateRequest):
        engine = _engine_holder.get("engine")
        if engine is None:
            return {"ok": False, "error": "No model loaded", "text": ""}
        try:
            from nanoforge.generation.sampling import SamplingConfig
            sc = SamplingConfig(mode=req.mode, temperature=req.temperature, top_k=req.top_k)
            return {"ok": True, "text": "".join(runner.stream(engine, req.prompt, req.max_new_tokens, sc))}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "text": ""}

    @app.post("/api/inference")
    def inference(req: GenerateRequest):
        engine = _engine_holder.get("engine")
        if engine is None:
            return {"ok": False, "error": "no model loaded"}
        task = getattr(getattr(engine.model, "config", None), "task_type", "causal_lm")
        try:
            if task in {"sequence_classification", "token_classification", "reward_modeling"} and hasattr(engine, "classify"):
                return {"ok": True, "task": task, "type": "classification", "labels": engine.classify(req.prompt)}
            from nanoforge.generation.sampling import SamplingConfig
            sc = SamplingConfig(mode=req.mode, temperature=req.temperature, top_k=req.top_k)
            text = "".join(runner.stream(engine, req.prompt, req.max_new_tokens, sc))
            return {"ok": True, "task": task, "type": "generation", "text": text}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ── Data ──────────────────────────────────────────────────────────────────
    @app.get("/api/data/list")
    def data_list(root: str = "data"):
        base = confined(root)
        if not base.exists():
            return {"ok": True, "files": []}
        files = []
        for count, p in enumerate(base.rglob("*")):
            if count >= 10000 or len(files) >= 500:
                break
            if p.is_file() and p.suffix.lower() in {".txt", ".jsonl", ".json", ".csv", ".tsv", ".parquet", ".bin"}:
                try:
                    files.append({"path": paths.relative(p), "size": p.stat().st_size})
                except (OSError, ValueError, PermissionError):
                    pass
        return {"ok": True, "files": files[:500]}

    @app.post("/api/data/profile")
    def data_profile(req: PathRequest):
        try:
            from nanoforge.data.profiler import profile_dataset
            prof = profile_dataset([confined(req.path)])
            return {"ok": True, "profile": json_safe(prof.to_dict())}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/data/catalog")
    def data_catalog():
        """Downloadable dataset cards for the web (ids match /api/jobs download)."""
        return {"ok": True, "catalog": _DATASET_CATALOG}

    @app.post("/api/data/upload")
    async def data_upload(file: UploadFile = File(...), subdir: str = Form("uploads")):  # noqa: B008
        """Commit a size-bounded upload under data/ only after complete receipt."""
        import tempfile
        original = file.filename or "upload.bin"
        from pathlib import PureWindowsPath
        if Path(original).name != original or PureWindowsPath(original).name != original or original in {".", ".."}:
            raise HTTPException(400, "Upload filename must be a plain filename.")
        safe_sub = confined(subdir, relative=True)
        dest_dir = confined(Path("data") / safe_sub.relative_to(paths.root))
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = confined(dest_dir / original)
        if dest.exists():
            raise HTTPException(409, "An upload with this filename already exists.")
        size = 0
        descriptor, temporary = tempfile.mkstemp(prefix=".upload-", suffix=".tmp", dir=dest_dir)
        try:
            with os.fdopen(descriptor, "wb") as out:
                while True:
                    chunk = await file.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > policy.max_request_bytes:
                        raise HTTPException(413, "Upload exceeds configured byte limit.")
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            confined(dest)
            try:
                os.link(temporary, dest)
            except FileExistsError as exc:
                raise HTTPException(409, "An upload with this filename already exists.") from exc
        except HTTPException:
            raise
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
            await file.close()
        return {"ok": True, "path": paths.relative(dest), "size": size}

    # ── Config ────────────────────────────────────────────────────────────────
    @app.get("/api/config/schema")
    def config_schema():
        """Full field schema for every config section — drives the param editor."""
        try:
            return {"ok": True, "schema": _config_schema()}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/registry")
    def registry():
        """All registry keys + metadata (attention backends, optimizers, heads, …)."""
        try:
            from nanoforge.registry import registry_snapshot
            return {"ok": True, "registries": json_safe(registry_snapshot())}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/config/build")
    def config_build(req: ConfigBuildRequest):
        """Validate section dicts into a NanoforgeConfig and return canonical YAML.

        Optionally writes the YAML to ``req.path``. Validation errors (bad registry
        keys, GQA divisibility, etc.) surface as ``ok: false`` with the message.
        """
        try:
            import dataclasses

            import yaml

            from nanoforge.config import parse_config
            s = req.sections
            cfg = parse_config(yaml.safe_dump(s, sort_keys=False), strict=True)
            yaml_text = yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False)
            written = None
            if req.path:
                out = confined(req.path)
                atomic_text(out, yaml_text)
                written = paths.relative(out)
            return {"ok": True, "yaml": yaml_text, "path": written}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/config/templates")
    def config_templates():
        base = confined("configs")
        cfgs = [paths.relative(p) for p in sorted(base.glob("*.yaml"))[:500]] if base.exists() else []
        return {"ok": True, "templates": cfgs}

    @app.get("/api/config/load")
    def config_load(path: str):
        try:
            from nanoforge.config import load_config
            return {"ok": True, "config": _flatten_config(load_config(confined(path)))}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/config/raw")
    def config_raw(path: str):
        from fastapi.responses import PlainTextResponse
        try:
            source = confined(path)
            if source.stat().st_size > policy.max_request_bytes:
                raise HTTPException(413, "Config exceeds byte limit.")
            return PlainTextResponse(source.read_text(encoding="utf-8"))
        except HTTPException:
            raise
        except Exception as exc:
            return PlainTextResponse(str(exc), status_code=400)

    @app.post("/api/config/save")
    def config_save(req: SaveConfigRequest):
        try:
            out = confined(req.path)
            from nanoforge.config import parse_config
            parse_config(req.yaml_text, strict=True)
            atomic_text(out, req.yaml_text)
            return {"ok": True, "path": paths.relative(out)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ── Models ────────────────────────────────────────────────────────────────
    @app.get("/api/model/list")
    def model_list():
        models = []
        for root in ("runs", "models"):
            base = confined(root)
            if base.exists():
                for count, p in enumerate(base.rglob("*.pt")):
                    if count >= 10000 or len(models) >= 200:
                        break
                    try:
                        models.append({"path": paths.relative(p), "ref": paths.relative(p), "kind": "native", "size": p.stat().st_size})
                    except (OSError, PermissionError, ValueError):
                        pass
        from nanoforge.imports import load_registry
        imported = load_registry(paths.root / "models/imports.json")
        models.extend({"path": entry.name, "ref": entry.name, "name": entry.name, "kind": "imported", "backend": entry.backend, "source": entry.source} for entry in imported.values())
        return {"ok": True, "models": models[:200]}

    @app.get("/api/model/imports")
    def model_imports():
        try:
            from nanoforge.imports import load_registry
            reg = load_registry(paths.root / "models/imports.json")
            return {"ok": True, "models": [{"name": k, **vars(v)} for k, v in reg.items()]}
        except Exception as exc:
            return {"ok": False, "models": [], "error": str(exc)}

    @app.post("/api/model/import")
    def model_import(req: ImportRequest):
        try:
            from nanoforge.imports import import_model
            source = confined(req.source)
            if not source.exists():
                raise ValueError("Imported studio models must exist in the selected workspace")
            if req.tokenizer:
                confined(req.tokenizer)
            entry = import_model(paths.relative(source), req.name, tokenizer=req.tokenizer, backend=req.backend, registry_file=paths.root / "models/imports.json")
            return {"ok": True, "entry": vars(entry)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/tasks")
    def list_tasks():
        from nanoforge.registry import TASK_HEADS, ensure_registry_ready
        ensure_registry_ready()
        return {"ok": True, "tasks": TASK_HEADS.names()}

    def project_info(value):
        from nanoforge.agent.project import (
            build_project_runtime,
            load_project_manifest,
            project_workspace,
        )
        source = confined(value)
        if source.stat().st_size > policy.max_request_bytes:
            raise HTTPException(413, "Project manifest exceeds byte limit")
        manifest = load_project_manifest(source)
        confined(project_workspace(manifest, source))
        workspace, registry, budget = build_project_runtime(manifest, source)
        return source, manifest, workspace, registry, budget

    @app.get("/api/workspace")
    def workspace_status():
        return {"ok": True, "workspace": ".", "run_dir": paths.relative(run_path), "resources": {"max_request_bytes": policy.max_request_bytes, "max_concurrency": policy.max_concurrency, "request_timeout": policy.request_timeout, "max_new_tokens": policy.max_new_tokens, "active_inference": runner.active}}

    @app.post("/api/runs/select")
    def select_run(req: PathRequest):
        nonlocal run_path
        run_path = confined(req.path)
        return {"ok": True, "run_dir": paths.relative(run_path)}

    @app.get("/api/projects/schema")
    def project_schema():
        from nanoforge.agent.project import _DEFAULT_BUDGETS, _SCHEMA_VERSION, ProjectManifest
        defaults = ProjectManifest(_SCHEMA_VERSION, "workspace-agent", ".", "manual", ("repo.list_files", "repo.read_file", "repo.search"), frozenset({"workspace.read"}), "read_only", dict(_DEFAULT_BUDGETS)).to_dict()
        return {"ok": True, "schema_version": _SCHEMA_VERSION, "defaults": defaults, "provider_kinds": ["manual", "native", "imported"], "approval_modes": ["read_only", "interactive"]}

    @app.get("/api/projects")
    def project_list():
        projects, skipped = [], 0
        candidates = list(paths.root.glob("*.yaml")) + list((paths.root / "configs").glob("*.yaml")) + list((paths.root / ".latticekin/projects").glob("*.yaml"))
        for source in candidates[:500]:
            try:
                _, manifest, _, _, _ = project_info(source)
                projects.append({"path": paths.relative(source), "name": manifest.name, "provider_kind": manifest.provider_kind, "fingerprint": manifest.fingerprint})
            except (ValueError, OSError, HTTPException):
                skipped += 1
        return {"ok": True, "projects": projects, "skipped": skipped}

    @app.post("/api/projects/load")
    def project_load(req: PathRequest):
        try:
            source, manifest, _, registry, _ = project_info(req.path)
            return {"ok": True, "path": paths.relative(source), "project": manifest.to_dict(), "tools": registry.schema(), "fingerprint": manifest.fingerprint}
        except (ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/projects/save")
    def project_save(req: ProjectSaveRequest):
        import tempfile

        import yaml
        target = confined(req.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".project-", suffix=".yaml", dir=target.parent)
        os.close(descriptor)
        try:
            Path(temporary).write_text(yaml.safe_dump(req.project, sort_keys=False), encoding="utf-8")
            _, manifest, _, registry, _ = project_info(temporary)
            atomic_text(target, yaml.safe_dump(manifest.to_dict(), sort_keys=False))
            return {"ok": True, "path": paths.relative(target), "project": manifest.to_dict(), "tools": registry.schema()}
        except (ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            Path(temporary).unlink(missing_ok=True)

    @app.get("/api/projects/context")
    def project_context(path: str):
        _, manifest, workspace, _, _ = project_info(path)
        files, used = [], 0
        limit = int(manifest.budgets.get("max_context_bytes", 262144))
        for value in manifest.resources["context_paths"]:
            remaining = min(workspace.max_file_bytes, limit - used)
            if remaining <= 0:
                break
            document = workspace.read_file(value, max_bytes=remaining)
            used += len(document["text"].encode("utf-8"))
            files.append(document)
        return {"ok": True, "files": files, "resources": {"context_bytes": used, "max_context_bytes": limit}, "truncated": len(files) < len(manifest.resources["context_paths"])}

    def agent_snapshot(run):
        with agent_lock:
            approvals = [{key: value for key, value in pending.items() if key not in {"event", "approved"}} for pending in run["pending"].values()]
            return {"run_id": run["run_id"], "project_path": run["project_path"], "state": run["state"], "done": run["done"], "error": run["error"], "result": run["result"], "pending_approvals": approvals, "resources": run["budget"].snapshot(), "trace_path": run["trace_path"]}

    @app.post("/api/agent/runs")
    def agent_start(req: AgentStartRequest):
        from dataclasses import asdict

        from nanoforge.agent.actions import decode_action
        from nanoforge.agent.harness import AgentHarness
        from nanoforge.agent.project import (
            build_project_provider,
            project_context_items,
            project_policy,
        )
        from nanoforge.agent.protocol import CancellationToken, InMemoryTrace
        source, manifest, workspace, registry, budget = project_info(req.project_path)
        if manifest.provider_kind == "manual" and not req.actions:
            raise HTTPException(422, "Manual projects require an explicit action sequence")
        for action in req.actions or ():
            try:
                decode_action(json.dumps(action, allow_nan=False))
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        run_id = uuid.uuid4().hex
        trace_file = confined(workspace.root / manifest.trace_dir / (run_id + ".json"))
        trace = InMemoryTrace(run_id, path=trace_file)
        cancellation = CancellationToken()
        run = {"run_id": run_id, "project_path": paths.relative(source), "state": "queued", "done": False, "error": None, "result": None, "pending": {}, "budget": budget, "trace": trace, "trace_path": paths.relative(trace_file), "cancellation": cancellation}
        with agent_lock:
            if len(agent_runs) >= 64:
                finished = next((key for key, value in agent_runs.items() if value["done"]), None)
                if finished is None:
                    raise HTTPException(429, "Agent run retention limit reached")
                agent_runs.pop(finished)
            agent_runs[run_id] = run

        class InteractiveApproval:
            def approve(self, call, spec):
                approval_id = uuid.uuid4().hex
                pending = {"approval_id": approval_id, "tool": spec.name, "arguments": call.arguments, "event": threading.Event(), "approved": False}
                with agent_lock:
                    run["pending"][approval_id] = pending
                    run["state"] = "awaiting_approval"
                try:
                    while not cancellation.cancelled and budget.remaining_seconds() > 0:
                        if pending["event"].wait(min(.1, budget.remaining_seconds())):
                            return pending["approved"]
                    return False
                finally:
                    with agent_lock:
                        run["pending"].pop(approval_id, None)
                        run["state"] = "running"

        def execute(cancel):
            cancellation._event = cancel
            run["state"] = "running"
            try:
                provider = build_project_provider(manifest, source, manual_actions=req.actions)
                capability_policy, approval = project_policy(manifest, approval=InteractiveApproval())
                context_items = project_context_items(manifest, workspace)
                result = AgentHarness(registry).run(req.request, provider, budget=budget, cancellation=cancellation, trace=trace, trace_path=trace_file, context_items=context_items, executor_kwargs={"policy": capability_policy, "approval": approval, "capabilities": manifest.capabilities})
                run["result"] = json_safe(asdict(result))
                run["state"] = result.status
                return {"run_id": run_id, "status": result.status, "trace_path": paths.relative(trace_file)}
            except Exception as exc:
                run["error"], run["state"] = str(exc), "failed"
                raise
            finally:
                run["done"] = True

        try:
            job_id = jobs.start_callable(execute, job_type="agent", run_dir=paths.relative(workspace.root))
        except RuntimeError as exc:
            agent_runs.pop(run_id, None)
            raise HTTPException(429, str(exc)) from exc
        run["job_id"] = job_id
        return {"ok": True, "run_id": run_id, "job_id": job_id}

    def find_agent(run_id):
        if run_id not in agent_runs:
            raise HTTPException(404, "Agent run not found")
        return agent_runs[run_id]

    @app.get("/api/agent/runs")
    def agent_list():
        return {"ok": True, "runs": [agent_snapshot(run) for run in reversed(list(agent_runs.values()))]}

    @app.post("/api/projects/evaluate")
    def project_evaluate(req: PathRequest):
        from nanoforge.agent.project import evaluate_project_model
        from nanoforge.agent.protocol import CancellationToken
        source, manifest, workspace, _, _ = project_info(req.path)
        if manifest.provider_kind == "manual" or not manifest.eval_suite:
            raise HTTPException(422, "Project evaluation requires a selected model and eval_suite")
        def evaluate(cancel):
            return evaluate_project_model(manifest, source, cancellation=CancellationToken(cancel))
        try:
            job_id = jobs.start_callable(evaluate, job_type="project_evaluation", run_dir=paths.relative(workspace.root))
        except RuntimeError as exc:
            raise HTTPException(429, str(exc)) from exc
        return {"ok": True, "job_id": job_id}

    @app.get("/api/agent/runs/{run_id}")
    def agent_status(run_id: str):
        return {"ok": True, "run": agent_snapshot(find_agent(run_id))}

    @app.get("/api/agent/runs/{run_id}/trace")
    def agent_trace(run_id: str):
        return {"ok": True, "events": find_agent(run_id)["trace"].to_list()}

    @app.post("/api/agent/runs/{run_id}/approve")
    def agent_approve(run_id: str, req: ApprovalRequest):
        run = find_agent(run_id)
        with agent_lock:
            pending = run["pending"].get(req.approval_id)
            if pending is None or pending["event"].is_set() or run["cancellation"].cancelled:
                raise HTTPException(409, "Approval is no longer pending")
            pending["approved"] = req.approved
            pending["event"].set()
        return {"ok": True}

    @app.post("/api/agent/runs/{run_id}/cancel")
    def agent_cancel(run_id: str):
        run = find_agent(run_id)
        run["cancellation"].cancel()
        jobs.cancel(run["job_id"])
        return {"ok": True}

    @app.post("/api/agent/replay")
    def agent_replay(req: ReplayRequest):
        from nanoforge.agent.harness import ReplayRunner
        if bool(req.path) == bool(req.run_id):
            raise HTTPException(422, "Provide exactly one run_id or trace path")
        value = confined(req.path) if req.path else find_agent(req.run_id)["trace"]
        return {"ok": True, "replay": ReplayRunner().replay(value)}

    @app.get("/api/evaluations/suites")
    def evaluation_suites():
        from nanoforge.evaluation.catalog import list_task_suites
        return {"ok": True, "suites": list_task_suites(paths.root)}

    @app.post("/api/evaluations/run")
    def evaluation_start(req: EvaluationRequest):
        suite, output = confined(req.suite_path), confined(req.output_dir)
        from nanoforge.evaluation.suite import load_task_suite
        load_task_suite(suite)
        model_ref = req.model_ref
        flag = "--checkpoint" if Path(model_ref).suffix.lower() in {".pt", ".pth"} else "--model"
        if flag == "--checkpoint":
            model_ref = str(confined(model_ref))
        else:
            from nanoforge.imports import load_registry
            entry = load_registry(paths.root / "models/imports.json").get(model_ref)
            local = confined(entry.source if entry else model_ref)
            if not local.exists():
                raise HTTPException(422, "Evaluation model must exist locally")
        command = [sys.executable, "-m", "nanoforge", "evaluate-suite", "--suite", str(suite), flag, model_ref, "--out", str(output)]
        try:
            job_id = jobs.start_process(command, job_type="evaluation", run_dir=paths.relative(output))
        except RuntimeError as exc:
            raise HTTPException(429, str(exc)) from exc
        return {"ok": True, "job_id": job_id}

    @app.get("/api/evaluations/reports")
    def evaluation_reports():
        from nanoforge.evaluation.catalog import list_evaluation_reports
        return {"ok": True, "reports": list_evaluation_reports(paths.root)}

    @app.get("/api/evaluations/report")
    def evaluation_report(path: str):
        from nanoforge.evaluation.catalog import load_evaluation_report
        return {"ok": True, **load_evaluation_report(confined(path))}

    @app.post("/api/evaluations/compare")
    def evaluation_compare(req: CompareRequest):
        from nanoforge.evaluation.catalog import compare_evaluation_reports
        return {"ok": True, "comparison": compare_evaluation_reports([confined(value) for value in req.report_paths])}

    @app.get("/api/artifacts")
    def artifacts():
        from nanoforge.artifacts import list_artifact_manifests
        return {"ok": True, "artifacts": list_artifact_manifests(paths.root)}

    @app.get("/favicon.ico")
    def favicon():
        from fastapi.responses import Response
        return Response(status_code=204)

    # ── Static SPA (must be last — catch-all would shadow API routes) ─────────
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="assets")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/{full_path:path}", response_class=HTMLResponse)
    def spa(full_path: str = ""):
        index = dist / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Run <code>npm run build</code> in the web/ directory first.</h1>", status_code=503)

    return app


def serve_webapp(run_dir: str = "runs/nanoforge", host: str = "127.0.0.1", port: int = 7860, *, api_key=None, workspace_root=None, max_request_bytes=2 * 1024 * 1024, max_concurrency=2, request_timeout=120.0, allowed_origins=None):
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Install serve extras: pip install -e .[serve]") from exc
    policy = ServingPolicy(host=host, api_key=api_key, max_request_bytes=max_request_bytes, max_concurrency=max_concurrency, request_timeout=request_timeout, allowed_origins=tuple(allowed_origins or ()))
    uvicorn.run(create_app(run_dir, workspace_root=workspace_root, policy=policy), host=host, port=port)
