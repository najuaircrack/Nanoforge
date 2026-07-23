# Changelog

All notable changes to Nanoforge are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow semantic
versioning once it reaches a stable release.

## [Unreleased]

### Added
- **Structure-aware dataset detection.** A new `profile_dataset` profiler
  (`src/nanoforge/data/profiler.py`) inspects records by *value shape*, not just field
  names, so nonstandard online shapes such as UltraChat `{"id","data":[turn, turn, ...]}`
  are recognized as chat conversations and trained with assistant-only loss masking.
- **`profile-dataset` CLI command** — preview detected mode, text columns, masking, turn
  field, and role style for any dataset (`--json` for the full profile).
- **`auto-train --mode auto`** now drives mode, text columns, and loss masking from the
  profiler and prints a one-line summary of what it understood.
- **Demo dataset + model.** `scripts/make_demo_dataset.py` produces a reproducible subset of
  raw UltraChat; `configs/demo-chat.yaml` trains a small (~14M-param) chat model end to end.
- **Graceful shutdown.** The trainer installs SIGINT/SIGTERM handlers, breaks at a safe step
  boundary, and saves an `interrupt.pt` emergency checkpoint before exiting.
- **Structured logging** via `src/nanoforge/logging_utils.py` (`get_logger`,
  `configure_logging`, level from `NANOFORGE_LOG_LEVEL`).
- **Config typo warnings.** Unknown keys in a config section are now logged (previously
  dropped silently) while still loading, so old configs keep working.
- **CI** (`.github/workflows/ci.yml`) running ruff + pytest on Python 3.10/3.11/3.12.
- `CONTRIBUTING.md` and this changelog.

### Changed
- `formats.py` recognizes bare turn lists (alternating strings or role dicts) as chat and
  formats them with `<|user|>` / `<|assistant|>` role markers.
- `configs/ultrachat-18m.yaml` run name, output dir, packed-data paths, and tokenizer path
  reconciled to the `ultrachat-18m` name used throughout the docs.
- `pyproject.toml` gained `[tool.ruff]` / `[tool.ruff.lint]` and
  `[tool.pytest.ini_options]` configuration.

### Fixed
- UltraChat-shaped data no longer trains fully masked. Because the old detector treated
  `{"id","data":[...]}` as generative text, packed labels were 100% `-100` and loss was
  identically zero; with value-based detection the assistant turns are unmasked (~30% of
  tokens) and loss trains normally.
