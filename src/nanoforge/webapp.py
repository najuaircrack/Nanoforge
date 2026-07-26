"""
Nanoforge Web App — full-scope FastAPI backend.
Serves the React SPA and provides all API endpoints.
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from nanoforge.progress import json_safe, read_jsonl_tail

try:
    from pydantic import BaseModel
except Exception:  # pydantic ships with fastapi; create_app raises a clean error if missing
    BaseModel = object  # type: ignore[assignment,misc]


# ── Request models (module level so FastAPI resolves them as request bodies) ──
class JobStartRequest(BaseModel):
    type: str
    config: str | None = None
    input: list[str] | None = None
    out: str | None = None
    mode: str = "auto"
    tokenizer: str = "byte"
    seq_len: int = 512
    dataset: str | None = None


class JobStopRequest(BaseModel):
    job_id: str


class LoadRequest(BaseModel):
    path: str


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.8
    top_k: int = 50
    max_new_tokens: int = 256
    mode: str = "balanced"


class PathRequest(BaseModel):
    path: str


class SaveConfigRequest(BaseModel):
    path: str
    yaml_text: str


class ImportRequest(BaseModel):
    source: str
    name: str
    backend: str | None = None
    tokenizer: str | None = None


class ChatStreamRequest(BaseModel):
    messages: list[list[str]]  # [[role, content], ...]
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 50


class RagLoadRequest(BaseModel):
    path: str
    top_k: int = 3


class FactsLoadRequest(BaseModel):
    path: str
    threshold: float = 0.5


class ConfigBuildRequest(BaseModel):
    sections: dict[str, dict[str, Any]]  # {"model": {...}, "training": {...}, ...}
    path: str | None = None  # if set, also write the built YAML here


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
    ("inference", "mode"): ["balanced", "creative", "precise", "greedy"],
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

# ── Job registry ──────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}  # job_id -> {proc, lines, done, code}
_jobs_lock = threading.Lock()


def _start_job(cmd: list[str]) -> str:
    job_id = str(uuid.uuid4())
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    entry: dict[str, Any] = {"proc": proc, "lines": [], "done": False, "code": None}
    with _jobs_lock:
        _jobs[job_id] = entry

    def _reader():
        for line in proc.stdout:
            line = line.rstrip()
            with _jobs_lock:
                entry["lines"].append(line)
        proc.wait()
        with _jobs_lock:
            entry["done"] = True
            entry["code"] = proc.returncode

    threading.Thread(target=_reader, daemon=True).start()
    return job_id


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


def _load_engine_compat(path: str):
    try:
        from nanoforge.generation.engine import GenerationEngine
        return GenerationEngine.from_checkpoint(path), None
    except Exception as exc:
        return None, str(exc)


# ── App factory ───────────────────────────────────────────────────────────────
def create_app(run_dir: str = "runs/nanoforge"):
    try:
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import HTMLResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except Exception as exc:
        raise RuntimeError("Install serve extras: pip install -e .[serve]") from exc

    app = FastAPI(title="Nanoforge")
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.jsonl"

    dist = Path(__file__).parent / "web_dist"

    # ── Metrics ───────────────────────────────────────────────────────────────
    @app.get("/api/metrics")
    def metrics(limit: int = 2000):
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
    def metrics_summary(limit: int = 2000):
        """Latest values + server-computed derived metrics (ETA, throughput, gaps)."""
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
        base = Path(root)
        if not base.exists():
            return {"ok": True, "runs": []}
        runs = []
        for p in sorted(base.iterdir()):
            if p.is_dir():
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
        if req.type in ("train", "pretrain", "pipeline"):
            if not req.config:
                return {"ok": False, "error": "config required"}
            cmd = [py, "-m", "nanoforge", req.type, "--config", req.config]
        elif req.type == "auto-train":
            if not req.config:
                return {"ok": False, "error": "config required"}
            cmd = [py, "-m", "nanoforge", "auto-train", "--config", req.config]
        elif req.type == "prepare":
            if not req.input or not req.out:
                return {"ok": False, "error": "input and out required"}
            cmd = [py, "-m", "nanoforge", "prepare",
                   "--input", *req.input, "--out", req.out,
                   "--mode", req.mode, "--tokenizer", req.tokenizer,
                   "--seq-len", str(req.seq_len)]
        elif req.type == "download":
            if not req.dataset:
                return {"ok": False, "error": "dataset required"}
            repo = _DATASET_REPOS.get(req.dataset, req.dataset)
            safe_name = req.dataset.replace("/", "__")
            cfg = f", {_DATASET_CONFIGS[req.dataset]!r}" if req.dataset in _DATASET_CONFIGS else ""
            cmd = [py, "-c",
                   "import sys; from datasets import load_dataset; "
                   f"ds=load_dataset({repo!r}{cfg}); "
                   f"ds.save_to_disk('data/hf/{safe_name}'); "
                   "print('done:', {k: len(v) for k, v in ds.items()})"]
        else:
            return {"ok": False, "error": f"unknown job type: {req.type}"}
        job_id = _start_job(cmd)
        return {"ok": True, "job_id": job_id}

    @app.post("/api/jobs/stop")
    def job_stop(req: JobStopRequest):
        with _jobs_lock:
            entry = _jobs.get(req.job_id)
        if entry and not entry["done"]:
            try:
                entry["proc"].terminate()
            except Exception:
                pass
        return {"ok": True}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        with _jobs_lock:
            entry = _jobs.get(job_id)
        if not entry:
            return {"ok": False, "error": "not found"}
        return {"ok": True, "done": entry["done"], "code": entry["code"], "lines": entry["lines"]}

    # ── SSE job stream ────────────────────────────────────────────────────────
    @app.get("/stream/job/{job_id}")
    def stream_job(job_id: str):
        def _gen():
            sent = 0
            import time
            while True:
                with _jobs_lock:
                    entry = _jobs.get(job_id)
                if entry is None:
                    yield f"data: {json.dumps({'type':'done','code':-1})}\n\n"
                    return
                lines = entry["lines"]
                while sent < len(lines):
                    yield f"data: {json.dumps({'type':'line','line':lines[sent]})}\n\n"
                    sent += 1
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
        engine, err = _load_engine_compat(req.path)
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
            retriever = load_index(req.path, engine=_engine_holder.get("engine"))
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
                req.path, engine=_engine_holder.get("engine"), threshold=req.threshold
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
        retriever = _rag_holder.get("retriever")
        if retriever is not None:
            query = next((c for r, c in reversed(messages) if r == "user"), "")
            if query:
                try:
                    hits = retriever.query(query, top_k=_rag_holder.get("top_k", 3))
                    if hits:
                        blocks = "\n\n".join(f"[{i+1}] {h['text_chunk']}" for i, h in enumerate(hits))
                        system = f"Relevant context:\n{blocks}"
                except Exception:
                    pass

        def _gen():
            try:
                from nanoforge.generation.sampling import SamplingConfig
                sc = SamplingConfig(mode="chat", temperature=req.temperature, top_k=req.top_k)
                for chunk in engine.chat_stream(
                    messages, max_new_tokens=req.max_new_tokens, sampling=sc, system=system
                ):
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
            return {"text": "[no model loaded]"}
        try:
            from nanoforge.generation.sampling import SamplingConfig
            sc = SamplingConfig(mode=req.mode, temperature=req.temperature, top_k=req.top_k)
            return {"text": engine.complete(req.prompt, max_new_tokens=req.max_new_tokens, sampling=sc) or "(empty)"}
        except Exception as exc:
            return {"text": f"[error: {exc}]"}

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
            text = engine.complete(req.prompt, max_new_tokens=req.max_new_tokens, sampling=sc)
            return {"ok": True, "task": task, "type": "generation", "text": text or "(empty)"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ── Data ──────────────────────────────────────────────────────────────────
    @app.get("/api/data/list")
    def data_list(root: str = "data"):
        base = Path(root)
        if not base.exists():
            return {"ok": True, "files": []}
        files = []
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".txt", ".jsonl", ".json", ".csv", ".tsv", ".parquet", ".bin"}:
                try:
                    files.append({"path": str(p), "size": p.stat().st_size})
                except OSError:
                    pass
        return {"ok": True, "files": files[:500]}

    @app.post("/api/data/profile")
    def data_profile(req: PathRequest):
        try:
            from nanoforge.data.profiler import profile_dataset
            prof = profile_dataset([req.path])
            return {"ok": True, "profile": json_safe(prof.to_dict())}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/data/catalog")
    def data_catalog():
        """Downloadable dataset cards for the web (ids match /api/jobs download)."""
        return {"ok": True, "catalog": _DATASET_CATALOG}

    @app.post("/api/data/upload")
    async def data_upload(file: UploadFile = File(...), subdir: str = Form("uploads")):  # noqa: B008
        """Stream an uploaded file to ``data/<subdir>/`` without loading it into RAM."""
        safe_name = os.path.basename(file.filename or "upload.bin").replace("\\", "_")
        safe_sub = os.path.normpath(subdir).replace("..", "").strip("/\\") or "uploads"
        dest_dir = Path("data") / safe_sub
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe_name
        size = 0
        try:
            with dest.open("wb") as out:
                while True:
                    chunk = await file.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    size += len(chunk)
                    out.write(chunk)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            await file.close()
        return {"ok": True, "path": str(dest), "size": size}

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

            from nanoforge.config import (
                DataConfig,
                InferenceConfig,
                ModelConfig,
                NanoforgeConfig,
                TrainConfig,
                _dataclass_from_dict,
            )
            s = req.sections
            cfg = NanoforgeConfig(
                model=_dataclass_from_dict(ModelConfig, s.get("model", {})),
                training=_dataclass_from_dict(TrainConfig, s.get("training", {})),
                data=_dataclass_from_dict(DataConfig, s.get("data", {})),
                inference=_dataclass_from_dict(InferenceConfig, s.get("inference", {})),
            )
            yaml_text = yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False)
            written = None
            if req.path:
                out = Path(req.path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(yaml_text, encoding="utf-8")
                written = str(out)
            return {"ok": True, "yaml": yaml_text, "path": written}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/config/templates")
    def config_templates():
        cfgs = [str(p) for p in sorted(Path("configs").glob("*.yaml"))] if Path("configs").exists() else []
        return {"ok": True, "templates": cfgs}

    @app.get("/api/config/load")
    def config_load(path: str):
        try:
            from nanoforge.config import load_config
            return {"ok": True, "config": _flatten_config(load_config(path))}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/config/raw")
    def config_raw(path: str):
        from fastapi.responses import PlainTextResponse
        try:
            return PlainTextResponse(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            return PlainTextResponse(str(exc), status_code=400)

    @app.post("/api/config/save")
    def config_save(req: SaveConfigRequest):
        try:
            out = Path(req.path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(req.yaml_text, encoding="utf-8")
            from nanoforge.config import load_config
            load_config(out)
            return {"ok": True, "path": str(out)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ── Models ────────────────────────────────────────────────────────────────
    @app.get("/api/model/list")
    def model_list():
        models = []
        for root in ("runs", "models"):
            base = Path(root)
            if base.exists():
                for p in sorted(base.rglob("*.pt")):
                    try:
                        models.append({"path": str(p), "size": p.stat().st_size})
                    except OSError:
                        pass
        return {"ok": True, "models": models[:200]}

    @app.get("/api/model/imports")
    def model_imports():
        try:
            from nanoforge.imports import load_registry
            reg = load_registry()
            return {"ok": True, "models": [{"name": k, **vars(v)} for k, v in reg.items()]}
        except Exception as exc:
            return {"ok": False, "models": [], "error": str(exc)}

    @app.post("/api/model/import")
    def model_import(req: ImportRequest):
        try:
            from nanoforge.imports import import_model
            entry = import_model(req.source, req.name, tokenizer=req.tokenizer, backend=req.backend)
            return {"ok": True, "entry": vars(entry)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.get("/api/tasks")
    def list_tasks():
        from nanoforge.registry import TASK_HEADS, ensure_registry_ready
        ensure_registry_ready()
        return {"ok": True, "tasks": TASK_HEADS.names()}

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


def serve_webapp(run_dir: str = "runs/nanoforge", host: str = "127.0.0.1", port: int = 7860):
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Install serve extras: pip install -e .[serve]") from exc
    uvicorn.run(create_app(run_dir), host=host, port=port)
