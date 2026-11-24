# Latticekin Changelog

All notable Latticekin changes are documented here. The Python compatibility namespace remains
`nanoforge` during the public-name migration.

## [Unreleased]

### November correctness and release modernization

- Added a source-only Docker context and archive exclusions for private/runtime state, dependency
  extras for imported adaptation/serving/export, portable offline checks and Linux/Windows,
  web, locked Rust, package and container CI jobs. CI configuration is not remote-run evidence.
- Added newly authored public C/chat fixtures and matching configs, a CPU train/save/held-out-eval
  smoke command, actual source-archive/wheel-install checks, and refusal to overwrite fixture outputs.
- Corrected the broken MoE schema/FFN preset, inconsistent recipes and model-origin/algorithm/
  streaming claims. Config and CLI references are generated from runtime definitions.
- Recorded subsystem regression repairs and remaining hardware/model/product gates in the roadmap.

### Platform foundation

- Restored the package metadata, documentation, license, and repository hygiene rules that are
  required for a clean checkout.
- Added the platform audit describing the current transformer toolkit, the agent-platform gap,
  CPU quality strategy, and an evidence-driven delivery sequence.
- Added `AGENTS.md` as the repository instruction ledger for safe, self-improving work.
- Added Phase 1 artifact/run manifests and the `evaluate-suite` task evaluator with explicit
  exact-match, structured JSON, tool-call, and lexical grounding checks.
- Added the first coding-agent slice: bounded repository tools, structured tool schemas,
  approval/capability checks, budgets, cancellation, replay traces, and deterministic FIM data
  formatting for code-specialist training.
- Added the portable `c-tiny` CPU/FIM fixture direction with a synthetic C corpus, a reproducible preparation
  workflow, and honest smoke-test limitations. Corrected FIM formatting so suffix context appears
  before the middle target instead of training only on a continuation after the suffix marker;
  The current FIM path uses fixed-length sequences, internal missing spans and held-out source groups.
- Defined the Latticekin public goal, first coding/workspace-agent wedge, anti-goals, custom TUI
  harness state machine, and staged CPU/GPU architecture in `docs/project-charter.md` and
  `docs/agent-platform-architecture.md`. Added `latticekin` as the public CLI/distribution name
  while retaining `nanoforge` as a compatibility alias.
- Added the first functional harness-console slice: strict project manifests, explicit manual
  structured actions, manifest tool allowlists and budgets, read-only capability enforcement,
  symlink confinement for repository discovery, and a Textual Harness screen with live trace output.
