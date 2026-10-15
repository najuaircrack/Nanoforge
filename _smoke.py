"""Import smoke check retained for maintainers; use pytest for behavior checks."""

import importlib

MODULES = [
    "nanoforge.cli",
    "nanoforge.config",
    "nanoforge.server",
    "nanoforge.webapp",
    "nanoforge.model.transformer",
    "nanoforge.data.dataset",
    "nanoforge.data.formats",
    "nanoforge.data.profiler",
    "nanoforge.data.tokenizer",
    "nanoforge.training.trainer",
    "nanoforge.training.pipeline",
    "nanoforge.training.checkpoint",
    "nanoforge.generation.engine",
    "nanoforge.generation.sampling",
    "nanoforge.evaluation.metrics",
    "nanoforge.export.onnx",
    "nanoforge.export.gguf",
]


def main() -> int:
    failures: list[str] = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    print(f"OK {len(MODULES) - len(failures)}/{len(MODULES)}")
    for failure in failures:
        print(f"FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
