import importlib, traceback

mods = [
    "nanoforge.cli",
    "nanoforge.config",
    "nanoforge.tui",
    "nanoforge.dashboard",
    "nanoforge.imports",
    "nanoforge.server",
    "nanoforge.progress",
    "nanoforge.templates",
    "nanoforge.benchmark",
    "nanoforge.profiling",
    "nanoforge.model.transformer",
    "nanoforge.data.dataset",
    "nanoforge.data.formats",
    "nanoforge.data.profiler",
    "nanoforge.data.tokenizer",
    "nanoforge.data.native_tokenizer",
    "nanoforge.data.tokenizer_metrics",
    "nanoforge.data.tokenizer_benchmark",
    "nanoforge.training.trainer",
    "nanoforge.training.pipeline",
    "nanoforge.training.checkpoint",
    "nanoforge.generation.engine",
    "nanoforge.generation.sampling",
    "nanoforge.evaluation.metrics",
    "nanoforge.export.onnx",
    "nanoforge.export.gguf",
]
ok, bad = [], []
for m in mods:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        bad.append((m, f"{type(e).__name__}: {e}"))
print(f"OK {len(ok)}/{len(mods)}")
for m, e in bad:
    print("FAIL", m, "->", e)
