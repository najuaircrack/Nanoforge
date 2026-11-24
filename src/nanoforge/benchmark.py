from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

import torch

from nanoforge.config import load_config
from nanoforge.generation.sampling import SamplingConfig
from nanoforge.model.transformer import NanoforgeForCausalLM
from nanoforge.profiling import estimate_model_profile
from nanoforge.training.utils import resolve_device


def process_memory_bytes() -> dict[str, int | str | None]:
    """Observed RSS and OS process-lifetime high-water mark; never an analytic estimate."""
    rss = peak = None
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            rss, peak = int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    else:
        try:
            import resource
            high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = int(high if sys.platform == "darwin" else high * 1024)
            if sys.platform.startswith("linux"):
                resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
                rss = resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (ImportError, OSError, ValueError):
            pass
    return {"rss_bytes": rss, "process_peak_rss_bytes": peak,
            "peak_rss_scope": "process_lifetime"}


@torch.inference_mode()
def benchmark_forward(config_path: str | Path, batch_size: int = 1, steps: int = 20,
                      device: str = "auto", *, cpu_threads: int | None = None,
                      warmup_steps: int = 3, seed: int = 1337) -> dict:
    if batch_size < 1 or steps < 1 or warmup_steps < 0 or (cpu_threads is not None and cpu_threads < 1):
        raise ValueError("Benchmark batch size, steps and CPU thread counts must be positive.")
    cfg = load_config(config_path)
    if not 1 <= cfg.data.seq_len <= cfg.model.max_seq_len:
        raise ValueError("Benchmark data sequence length must fit the model context.")
    dev = resolve_device(device)
    previous_threads = torch.get_num_threads()
    if cpu_threads is not None:
        if dev.type != "cpu":
            raise ValueError("CPU thread tuning requires a CPU benchmark.")
        torch.set_num_threads(cpu_threads)
    try:
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(seed)
            model = NanoforgeForCausalLM(cfg.model).to(dev).eval()
            x = torch.randint(0, cfg.model.vocab_size, (batch_size, cfg.data.seq_len), device=dev)
            for _ in range(warmup_steps):
                model(x)
            before = process_memory_bytes()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            started = time.perf_counter()
            for _ in range(steps):
                model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            elapsed = time.perf_counter() - started
            after = process_memory_bytes()
            estimated = estimate_model_profile(cfg.model, batch_size=batch_size,
                                                seq_len=cfg.data.seq_len).to_dict()
            return {
                **estimated,  # retained compatibility; nested field explicitly labels their origin
                "estimated_profile": estimated,
                "params": sum(parameter.numel() for parameter in model.parameters()),
                "tokens_per_sec": batch_size * cfg.data.seq_len * steps / elapsed,
                "ms_per_step": elapsed * 1000 / steps,
                "elapsed_seconds": elapsed, "steps": steps, "warmup_steps": warmup_steps,
                "batch_size": batch_size, "seq_len": cfg.data.seq_len, "seed": seed,
                "device": str(dev), "cpu_threads": torch.get_num_threads(),
                "torch_version": torch.__version__,
                "rss_before_bytes": before["rss_bytes"],
                "rss_after_bytes": after["rss_bytes"],
                "process_peak_rss_bytes": after["process_peak_rss_bytes"],
                "peak_rss_scope": "process_lifetime",
                "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None),
                "actual_attention_backends": [block.attn.actual_backend for block in model.backbone.blocks],
                "quality_status": "not_evaluated",
            }
    finally:
        if cpu_threads is not None:
            torch.set_num_threads(previous_threads)


@torch.inference_mode()
def benchmark_generation(engine, prompt: str, max_new_tokens: int = 32, *,
                         repeats: int = 3, sampling: SamplingConfig | None = None,
                         warmup_steps: int = 1, seed: int = 1337) -> dict:
    if max_new_tokens < 1 or repeats < 1 or warmup_steps < 0:
        raise ValueError("Generation benchmark token/repeat counts must be positive.")
    sampling = sampling or SamplingConfig(mode="deterministic", stop_on_repetition=False)
    ids = engine.tokenizer.encode(prompt, add_bos=True, add_eos=False)
    runs = []
    devices = [engine.device] if engine.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for _ in range(warmup_steps):
            list(engine.generate_ids(ids, max_new_tokens, sampling, eos_id=engine.tokenizer.eos_id))
        before = process_memory_bytes()
        if engine.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(engine.device)
        for _ in range(repeats):
            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
            started = time.perf_counter()
            first = None
            generated = []
            for token in engine.generate_ids(ids, max_new_tokens, sampling, eos_id=engine.tokenizer.eos_id):
                if first is None:
                    if engine.device.type == "cuda":
                        torch.cuda.synchronize(engine.device)
                    first = time.perf_counter() - started
                generated.append(token)
            if engine.device.type == "cuda":
                torch.cuda.synchronize(engine.device)
            elapsed = time.perf_counter() - started
            runs.append({"generated_tokens": len(generated), "elapsed_seconds": elapsed,
                         "first_token_seconds": first, "output": engine.tokenizer.decode(generated)})
        after = process_memory_bytes()
    total_tokens = sum(run["generated_tokens"] for run in runs)
    elapsed = sum(run["elapsed_seconds"] for run in runs)
    return {"schema_version": 1, "benchmark": "autoregressive_generation", "device": str(engine.device),
            "seed": seed, "cpu_threads": torch.get_num_threads(), "runs": runs,
            "tokens_per_sec": total_tokens / elapsed, "generated_tokens": total_tokens,
            "rss_before_bytes": before["rss_bytes"], "rss_after_bytes": after["rss_bytes"],
            "process_peak_rss_bytes": after["process_peak_rss_bytes"], "peak_rss_scope": "process_lifetime",
            "prefix_cache_bytes": engine.prefix_cache_bytes,
            "cuda_peak_allocated_bytes": (torch.cuda.max_memory_allocated(engine.device)
                                          if engine.device.type == "cuda" else None),
            "actual_attention_backends": [block.attn.actual_backend for block in engine.model.backbone.blocks],
            "quality_status": "not_evaluated"}
