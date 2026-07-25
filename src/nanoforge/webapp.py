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
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except Exception as exc:
        raise RuntimeError("Install serve extras: pip install -e .[serve]") from exc

    app = FastAPI(title="Nanoforge")
    run_path = Path(run_dir)
    metrics_path = run_path / "metrics.jsonl"
    _engine_holder: dict = {"engine": None, "path": None}

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

    @app.post("/api/chat/stream")
    def chat_stream_sse(req: ChatStreamRequest):
        engine = _engine_holder.get("engine")
        if engine is None:
            def _err():
                yield f"data: {json.dumps({'type':'error','text':'No model loaded'})}\n\n"
            return StreamingResponse(_err(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        messages = [(r, c) for r, c in req.messages]
        def _gen():
            try:
                for chunk in engine.chat_stream(messages, max_new_tokens=req.max_new_tokens):
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

    # ── Config ────────────────────────────────────────────────────────────────
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


# Keep backward-compat alias
def serve_dashboard(run_dir, host: str = "127.0.0.1", port: int = 7860):
    serve_webapp(run_dir, host, port)
