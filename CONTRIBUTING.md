# Contributing to Nanoforge

Nanoforge is moving from a compact transformer experiment into a local-first platform for
training, adapting, evaluating, and safely running small agent models. Keep changes narrow,
observable, and reproducible.

## Development setup

Nanoforge targets Python 3.10–3.12; `.python-version` records the default development runtime.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The web application has its own lockfile and toolchain:

```bash
cd web
npm ci
npm run build
npm run lint
```

## Checks

Run the focused offline project check from the repository root:

```bash
make test
```

The underlying checks are:

```bash
python -m ruff check .
python -m pytest -q
```

When changing the web app, also run `npm run build` and `npm run lint` from `web/`.

## Change rules

1. Read `AGENTS.md` and the relevant architecture document before editing.
2. Add a regression test for behavior changes, especially data masking, checkpoint loading,
   generation stopping, tool execution, or security boundaries.
3. Do not commit datasets, model weights, private conversations, indexes, caches, or secrets.
4. Do not describe placeholder or fallback implementations as production backends.
5. Record user-visible changes in `CHANGELOG.md` and update the relevant workflow documentation.
6. Run the smallest meaningful check after each working step; keep unrelated failures visible.

## Project layout

- `src/nanoforge/` — Python package and public runtime APIs.
- `web/` — React/Vite local studio.
- `configs/` — checked-in, portable configuration examples.
- `docs/` — architecture, workflows, platform audit, and roadmap.
- `native/` — optional Rust preprocessing/tokenizer extension.
- `tests/` — executable behavior checks.
- `scripts/` — small, documented developer utilities only.
