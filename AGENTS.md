# Latticekin agent instruction ledger

This file is the operating contract for coding agents working in this repository. It is
deliberately evidence-first: an agent may improve the system, but it must not silently change
the product, hide a failed check, or turn an experimental fallback into a claimed feature.

## Mission

Latticekin is a local-first platform for people to:

1. inspect and clean their own data;
2. train or adapt a language model on CPU, one GPU, or a small distributed job;
3. evaluate the model against repeatable tasks;
4. compose the model with retrieval, memory, tools, and an explicit agent harness;
5. export and serve the result locally with clear resource and safety boundaries.

The current repository is **not yet the complete platform**. It is a compact transformer
training/inference toolkit with a partial web studio, a bounded backend harness, and a training
TUI that is not yet the complete agent operator console. Treat `docs/project-charter.md`,
`docs/agent-platform-architecture.md`, and `docs/platform-audit.md` as the current product
contract and backlog, not as evidence that all listed capabilities already exist.

## Read before editing

1. Read this file and the relevant section of `docs/platform-audit.md`.
2. Inspect `git status --short` and preserve unrelated work.
3. Find the real caller, public API, saved format, and test before changing an implementation.
4. Read the nearest project documentation and existing tests; do not infer behavior from names.
5. State the smallest observable outcome for the change before writing code.

## Source-of-truth boundaries

- **Runtime behavior:** `src/nanoforge/` and executable tests in `tests/`.
- **Web behavior:** `web/src/`, `web/package.json`, `web/package-lock.json`, and its build.
- **Portable examples:** `configs/`, `docs/`, and documented scripts under `scripts/`.
- **Native acceleration:** `native/nanoforge-tokenizers/`; Python fallbacks remain mandatory.
- **Private/runtime state:** `data/`, `models/`, `runs/`, `reports/`, caches, and personal files.
  Never copy private data into source, tests, docs, logs, prompts, or agent memory.
- **Architecture decisions:** `docs/platform-audit.md` and `docs/roadmap.md`, updated only when
  the implementation or an explicit product decision changes them.

## Non-negotiable engineering rules

### Reproducibility

- Keep paths repository-relative. Never commit machine-specific absolute paths.
- Keep `.python-version`, `pyproject.toml`, `web/package-lock.json`, and Rust `Cargo.lock`
  aligned with the supported workflow.
- Keep a working `Dockerfile`, `.dockerignore`, `Makefile`, README setup steps, and one offline
  test command. Do not claim a clean build until it was actually run.
- Make random seeds, data fingerprints, tokenizer identity, model config, and training settings
  part of artifacts that affect reproducibility.
- Prefer atomic writes, explicit schema versions, and migration tests for saved artifacts.

### Quality and model honesty

- Do not optimize for training loss alone. A model is useful only when held-out loss, task
  evaluations, generation behavior, latency, memory, and failure cases are measured together.
- Do not call a registry alias a real implementation. Names such as `paged`, `sparse`, `flash`,
  `mamba_selective_state`, and `rwkv_mixing` must retain honest metadata until their kernels or
  algorithms are genuinely implemented.
- Prefer a good pretrained base plus data-quality-controlled adaptation over training a tiny
  foundation model from scratch when a user expects broad knowledge.
- Use CPU-friendly dense models, GQA/MQA, tied embeddings, good tokenization, gradient
  accumulation, checkpointing, and quantized inference before adding exotic architecture.
- Keep the byte tokenizer for smoke tests, not as the default quality path for real language
  training.
- Never silently drop invalid config keys, labels, records, or tool calls. Warn, count, and make
  the outcome inspectable.

### Agent-platform safety

- Tool execution must have a schema, validation, timeout, cancellation path, output limit,
  audit event, and explicit approval policy. Never execute arbitrary model-generated shell code
  by default.
- Agent memory must declare scope, owner, retention, provenance, and deletion behavior. Personal
  facts are opt-in and must remain separate from generic training data.
- Network, filesystem, subprocess, and credential access are capabilities, not defaults.
- Localhost-only is the default for servers. Any host exposure requires authentication,
  authorization, request limits, and a security test.
- Do not add autonomous self-modification. A self-improving loop may propose changes, run
  evaluations, and create a reviewable patch; only a human-approved change may alter source or
  policy.

## Self-improvement loop

Every meaningful improvement follows this ledger:

1. **Observe** — capture a reproducible baseline, including the exact command, environment,
   inputs, metrics, and known failures.
2. **Diagnose** — identify the limiting mechanism and separate evidence from hypothesis.
3. **Propose** — write a small change, acceptance criteria, risk, rollback path, and affected
   files. Prefer one layer at a time: data, model, training, evaluation, harness, or web UI.
4. **Test** — add a focused regression or fixture before changing behavior when practical.
5. **Implement** — make the smallest complete change; update callers, schemas, docs, and logs
   together.
6. **Evaluate** — run unit, integration, smoke, quality, and performance checks appropriate to
   the change. Compare to the baseline rather than relying on intuition.
7. **Record** — update the changelog, audit/roadmap status, and handoff with actual results.
8. **Review** — inspect the final diff for secrets, private data, dead files, portability gaps,
   misleading claims, and unverified gates.

An agent must stop and report a gap when a required check cannot run. It must not replace a
failing check with a weaker no-op merely to make the ledger green.

## Required workflows

### Data to model

`inspect-dataset` → `validate-dataset` → `profile-dataset` → clean/deduplicate → tokenizer
report → prepare/pack → validate label balance → train. Preserve a manifest containing source
fingerprints, filtering counts, tokenizer identity, sequence length, masking policy, and split
seed.

### Model quality

Start with a tiny end-to-end smoke fixture. Then run a CPU budgeted experiment with a held-out
split, fixed seed, sample prompts, task checks, and a resource report. For agent use, add tool
call validity, refusal/safety, retrieval faithfulness, and multi-turn tests. A loss decrease is
necessary but not sufficient.

### Agent harness

The target flow is `request → policy → context/memory → model → structured action → tool
validation/approval → observation → state update → final answer`. Each transition must be
traceable, bounded by a step/time/token budget, and replayable from a redacted trace. The harness
must work with an imported quantized model as well as a Nanoforge-trained checkpoint.

### Release

Run `make test`, web build/lint when web files changed, native `cargo test --locked` when Rust
files changed, then review `git diff --check`, `git status --short`, and the complete diff.
Generated outputs and private personal artifacts stay out of commits.

## Definition of done for future work

A change is done only when:

- the requested behavior exists end to end;
- the public path and failure path are tested;
- configuration and saved artifacts are backward-compatible or explicitly migrated;
- CPU, memory, latency, and quality trade-offs are measured where relevant;
- docs and examples match the implementation;
- no placeholder, stub, fake backend, secret, private dataset, or generated cache was added;
- the real checks were run and their results are recorded.

When scope is too large for one change, split it into vertical slices and leave the next slice
in `docs/platform-audit.md` or a project-local handoff. Do not claim the whole platform is done
because the foundation or UI exists.
