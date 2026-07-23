# Contributing to Nanoforge

Thanks for your interest in Nanoforge. This guide covers local setup, the test/lint
workflow, and how to open a pull request.

## Development setup

Nanoforge targets Python 3.10–3.12. Create a virtual environment and install the package
in editable mode with the dev extras:

```bash
python -m venv .venv
# Windows (Git Bash): source .venv/Scripts/activate
# macOS/Linux:        source .venv/bin/activate
pip install -e ".[dev]"
```

The `dev` extra pulls in `pytest` and `ruff`. For data/tokenizer/serving work, install the
matching extras (`.[data]`, `.[tokenizers]`, `.[serve]`) or `.[all]`.

## Tests and linting

Run both before opening a PR — CI runs the same commands on py3.10/3.11/3.12:

```bash
ruff check .
pytest -q
```

Configuration for both lives in `pyproject.toml` (`[tool.ruff]` / `[tool.ruff.lint]` and
`[tool.pytest.ini_options]`). Tests that need `torch` skip automatically when it isn't
installed, so a minimal `.[dev]` environment still collects green.

## Pull request flow

1. Branch off `main`.
2. Keep changes focused; match the surrounding code style (ruff enforces import order and
   pycodestyle rules; line length is advisory).
3. Add or update tests for behavior changes. New dataset shapes should get a case in
   `tests/test_profiler.py` or `tests/test_formats_detection.py`.
4. Update `CHANGELOG.md` under `## [Unreleased]`.
5. Ensure `ruff check .` and `pytest -q` pass, then open the PR against `main`.

## Project layout

- `src/nanoforge/` — package (data pipeline, model, training, generation, serving).
- `configs/` — training/inference configs.
- `scripts/` — standalone diagnostics and dataset tooling (not imported by the package).
- `tests/` — pytest suite.
- `docs/` — user-facing guides.
