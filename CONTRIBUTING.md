# Contributing to Latticekin

Read `AGENTS.md` and the relevant subsystem documentation before editing. Keep changes observable,
preserve unrelated work, and use public synthetic fixtures instead of personal data.

Use Python 3.10–3.12 and Node 24:

```text
python -m venv .venv
python -m pip install -e ".[dev,tokenizers,serve,tui,pretrained,export]"
```

Activate the environment before installing. Run `npm ci` from `web/` once; the Python suite
contains genuine frontend workflow checks and requires installed Node/web dependencies.

```text
python scripts/check.py
python scripts/check.py --web
python scripts/check.py --native
python scripts/check.py --package
```

`make test` delegates to the portable Python command. Offline checks do not install dependencies
or download models. Native offline checks need Rust dependencies fetched beforehand and the Python
runtime on PATH; on Windows the check script adds its interpreter directory for DLL discovery.
Install the native extension separately with Maturin or pip from `native/nanoforge-tokenizers/`;
the native extra installs build tooling.

Add regression tests for public behavior and failure paths. Keep checkpoints and saved formats
backward-compatible or provide explicit migrations. Record actual tests, quality/resource tradeoffs,
remaining gates and user-visible changes. Never replace a failed required check with a no-op.

Docker uses a source allowlist; release archives exclude runtime/private directories. Do not commit
datasets, weights, private conversations, indexes, traces, credentials or caches.
Treat algorithm aliases/approximations honestly. A passing synthetic test does not establish
GPU/FSDP performance, model quality, comprehensive privacy filtering or production multi-user isolation.
