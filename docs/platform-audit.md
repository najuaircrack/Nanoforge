# Nanoforge platform audit and modernization contract

**Status:** baseline audit, October 2026
**Audience:** maintainers and coding agents
**Decision:** keep the compact PyTorch/Rust foundation, but stop treating it as a complete agent
platform. Build the platform in measured vertical slices.

## Executive conclusion

Nanoforge is not an empty or useless project. It already contains a readable local transformer
stack: schema-aware data readers, tokenizers, masked packing, a configurable transformer,
checkpointing, sampling, RAG primitives, a FastAPI server, a React studio, DPO/distillation
experiments, and an optional Rust tokenizer extension.

It is also not currently able to create a generally capable agent by training a small model from
scratch on an ordinary CPU. The limiting factor is not one missing attention trick. It is the
whole product loop:

```text
data quality → pretrained capability → efficient adaptation → evaluation →
structured tool use → guarded execution → memory/retrieval → replayable deployment
```

The current repository is strongest in the first half of a research toolkit and weakest in
capability provenance, quality evaluation, CPU-efficient model serving, and the safe agent
harness. The correct strategy is therefore **adapt and compose first, train foundations second**.

## Current architecture

### Repository map

```text
Nanoforge/
├── src/nanoforge/
│   ├── cli.py                    # argparse commands and workflow orchestration
│   ├── config.py                 # dataclass config and YAML loading
│   ├── registry.py               # lazy registries and plugin discovery
│   ├── model/                    # transformer, attention, RoPE, FFN, heads, LoRA
│   ├── data/                     # readers, cleaning, tokenizers, packing, memmaps
│   ├── training/                 # trainer, DPO, distillation, checkpoint, health, DDP/FSDP
│   ├── generation/               # sampling, KV cache, streaming, chat formatting
│   ├── rag/                      # chunking, embeddings, TF-IDF/model indexes, retrieval
│   ├── evaluation/               # loss/perplexity/token metrics
│   ├── export/                   # GGUF and ONNX writers
│   ├── imports.py                # external GGUF/HF/ONNX model adapters
│   ├── server.py                 # local OpenAI-like inference API
│   ├── webapp.py                 # local training studio API and SPA host
│   └── tools.py                  # minimal function registry, not yet a harness
├── web/                          # React 19/Vite/TanStack/TypeScript studio
├── configs/                      # YAML model and training examples
├── docs/                         # workflows and architecture notes
├── tests/                        # pytest behavior and smoke checks
├── native/nanoforge-tokenizers/  # optional Rust/PyO3 tokenizer and cleaning path
└── scripts/                      # developer and data utilities; private scripts excluded
```

### Existing runtime flow

```text
raw files or HF refs
  → formats.iter_dataset_records()
  → cleaning / deduplication / profiling
  → tokenizer training or loading
  → packing.py writes token and optional label shards + manifests
  → PackedMemmapDataset samples windows
  → Trainer builds NanoforgeModel and runs PyTorch optimization
  → checkpoint + metrics.jsonl + samples
  → GenerationEngine / server / web studio
```

The CLI already exposes dataset inspection, validation, profiling, cleaning, deduplication,
conversion, tokenizer training and reporting, packing, training, pretraining, pipeline recipes,
DPO, distillation, evaluation, benchmarking, import/export, RAG indexing, serving, chat, TUI,
and web commands. The presence of a command is not proof that its production contract is complete.

## What works today

The following are real foundations, subject to their tests and documented limitations:

- PyTorch transformer training on CPU or one GPU, with gradient accumulation, checkpointing,
  gradient clipping, health metrics, atomic checkpoint files, and graceful interruption.
- GQA/MQA, tied embeddings, RMSNorm/LayerNorm, RoPE, SwiGLU/GEGLU, optional MoE, LoRA layers,
  SDPA/manual attention, and KV caching.
- Streaming-ish dataset ingestion across common text and structured formats, record cleaning,
  chat/instruction masking, fixed-sequence boundary packing, and memmap-backed sampling.
- Byte tokenizer for dependency-free smoke tests plus BPE/WordPiece/SentencePiece adapters and
  a Python fallback for the optional Rust tokenizer.
- DPO and knowledge-distillation training paths as experiments, not yet as a complete alignment
  evaluation system.
- Pure NumPy TF-IDF retrieval and optional model/sentence-transformer embeddings.
- GGUF/ONNX export paths, with int8 dynamic CPU quantization implemented and int4/GPTQ/AWQ
  explicitly rejected rather than faked.
- Local server and web studio defaults that can be run on localhost.

## Critical limitations

### 1. It is a scratch-model toolkit, not a capable-model factory

The default path initializes a new randomly initialized transformer. A 10–50M parameter model
trained on a small personal corpus can learn formatting and local style, but it cannot acquire
the broad language knowledge, reasoning, tool semantics, and robustness users associate with a
modern agent. Training loss falling on a narrow corpus is not evidence of general capability.

**Implication:** add first-class pretrained-model import and adapter workflows. Use Nanoforge to
prepare data, apply LoRA/QLoRA where supported, evaluate, compose tools/RAG, and export. Retain
from-scratch training for education, domain-specific small models, and controlled research.

### 2. CPU support is mostly “can run”, not “good quality per hour”

The model executes in PyTorch eager mode and CPU training forces fp32. The built-in quantization
path is dynamic int8 for CPU `Linear` layers; int4/GPTQ/AWQ are not implemented. The runtime does
not yet have a native quantized transformer inference engine, weight-only kernels, token-budget
batching, or a mature CPU performance report.

The data path also samples Python/NumPy windows and copies each batch into new tensors. This is
simple and testable, but it leaves CPU throughput on the table. `DataConfig.batch_size` is a
config/API concept while the trainer actually uses `micro_batch_size × grad_accum_steps` for
the effective batch, so configuration semantics need consolidation.

### 3. The quality loop is incomplete

Evaluation currently centers on loss, perplexity, token accuracy, and forward benchmarks. It does
not yet provide a versioned task suite for instruction following, tool-call validity, grounded
retrieval, refusal behavior, multi-turn consistency, or regression comparisons. There is no
automatic quality gate that can stop a “lower loss, worse assistant” change.

Every training run needs a manifest containing code revision, config, data fingerprints, split
policy, tokenizer identity, model size, seed, hardware, elapsed time, peak memory, and evaluation
results. Reports must preserve failed cases, not only aggregate scores.

### 4. There is no real agent harness yet

`src/nanoforge/tools.py` stores a name, description, JSON-like parameter dictionary, and Python
callable, then calls the callable directly. It has no schema validation, authorization, timeout,
cancellation, output truncation, audit event, approval policy, sandbox, replay trace, or
multi-step state machine. `webapp.py` can start subprocess jobs and write local paths; it is a
local admin UI, not a multi-user control plane.

The required target is:

```text
request
  → policy decision
  → bounded context and memory selection
  → model inference
  → structured action parsing
  → schema validation
  → approval / capability check
  → bounded tool execution
  → redacted observation
  → state update and trace
  → final answer or next step
```

No model-generated shell command, arbitrary Python, network request, filesystem write, or
credential access should be executable by default.

### 5. Memory and retrieval are primitives, not governed product features

RAG indexing and personal-facts helpers exist, but memory scope, owner, provenance, retention,
deletion, consent, and retrieval faithfulness are not represented as a common artifact contract.
Personal data is present in this worktree as ignored runtime material; it must never become a
fixture, test input, prompt example, committed index, or model artifact.

### 6. Registry names overstate implementation in places

The registry honestly annotates some fallbacks, but several names still look more advanced than
their implementation:

- `paged` is contiguous KV cache plus a full/local window, not PagedAttention.
- `sparse`, `chunked`, and `hybrid_local_global` are windowed SDPA presets, not sparse/chunked
  kernels or local/global schedules.
- `flash*` falls back to SDPA unless the external FlashAttention package and compatible CUDA
  runtime are present.
- `mamba_selective_state` and `rwkv_mixing` are experimental FFN mixers, not full architectures.
- sampler registry entries point to one sampling function; registry metadata does not yet define
  a complete sampler interface.

Future work must either implement the algorithm or mark the entry experimental/unavailable and
test the actual fallback.

### 7. Training correctness and distributed behavior need hardening

The base trainer is useful, but there are gaps to close before claiming production training:

- FSDP checkpoint saving does not yet consolidate a full state dict on rank zero.
- DDP gradient accumulation does not yet use `no_sync()` for intermediate microsteps.
- DPO preloads all preference pairs and does not yet share the base trainer's full health,
  evaluation, resume, distributed, and trace contracts.
- Trainer metric logging and sampling perform synchronous operations that can perturb throughput.
- Data splits are random record decisions rather than a named, persisted, leakage-resistant split
  strategy with group/document boundaries.
- Config unknown keys warn but are still ignored; strict mode is needed for release workflows.

### 8. The repository had a reproducibility break

At audit start, tracked root files including `pyproject.toml`, `README.md`, `.gitignore`, and the
license were deleted in the worktree, while the frontend manifests were untracked and the old
ignore file would have ignored nested `web/package.json`. Generated caches, Rust `target/`, egg
metadata, reports, and private data were present. The foundation files have now been restored and
the ignore rules corrected. Runtime/private artifacts remain outside the source contract.

## Research conclusions for good CPU models

These conclusions are grounded in the cited papers and implementation evidence, not optimism:

1. **Do not train broad knowledge from scratch on a personal CPU.** Scaling-law work shows that
   model size, data, and compute must be allocated together. Chinchilla-style compute-optimal
   training makes the data requirement explicit; a tiny model plus a tiny corpus is a specialist,
   not a general assistant.
2. **Data quality can make very small models useful in a narrow domain.** TinyStories shows that
   carefully controlled data can produce coherent small models. Nanoforge should ship clean,
   license-aware, domain-specific fixtures and evaluate the intended task rather than promise
   general intelligence.
3. **The practical CPU product is adaptation plus retrieval.** Start with a capable imported
   base model and adapt with LoRA/QLoRA or distillation where the hardware permits. Keep facts in
   retrieval/memory with citations instead of trying to memorize a changing private corpus.
4. **Architecture matters below one billion parameters.** MobileLLM reports that deep/thin
   designs, embedding sharing, GQA, and careful small-model architecture improve sub-billion
   quality. Nanoforge already has tied embeddings and GQA; future presets should be benchmarked
   rather than guessed.
5. **Inference is a separate engineering problem.** llama.cpp demonstrates the value of native
   CPU SIMD kernels, multiple quantization levels, CPU/GPU hybrid execution, and a dedicated
   serving runtime. Python should orchestrate and evaluate; it should not be the only hot path.
6. **Alignment data must be small but excellent.** LIMA and LoRA research support the strategy of
   using a strong base and carefully curated instruction/adaptation data. Quality, formatting,
   coverage, and held-out evaluation matter more than collecting noisy personal text.
7. **Distillation is a viable CPU bridge.** A larger teacher can produce high-quality targets and
   a smaller student can be exported for local use, but the distillation dataset and task checks
   must be explicit. A teacher copied from the student is not meaningful capability distillation.

### Recommended model tiers

| Tier | Purpose | Training strategy | CPU expectation |
| --- | --- | --- | --- |
| Smoke | pipeline correctness | byte tokenizer, tiny random model | seconds/minutes; not quality |
| Specialist | narrow local task | 20–150M dense model, good BPE, curated data | usable for constrained style/classification |
| Adapter | personal/domain assistant | imported 0.5–3B quantized base + LoRA/QLoRA or external runtime | best quality per local compute |
| Student | compact deployment | teacher-generated data + distillation + quantized export | lower latency, bounded domain |
| Foundation | broad model from scratch | out of scope for ordinary CPU; requires large data/compute | do not promise |

## Agent platform target

Nanoforge should become a **local agent forge**, not merely a model trainer. A model is one
replaceable component. The product contract should include:

### Project and artifact model

- A project has a repository-relative manifest with model/provider, tokenizer, data sources,
  eval suite, tools, memory stores, policies, budgets, and export targets.
- Every derived artifact has a schema version, content fingerprint, parent artifact IDs, owner,
  retention policy, and redacted provenance.
- A run is immutable after completion except for a human-authored annotation; results are
  comparable across model, data, prompt, harness, and runtime changes.

### Model lifecycle

```text
import or initialize
  → inspect / validate / clean data
  → tokenizer or base-model compatibility check
  → pretrain, SFT, LoRA/QLoRA, DPO, or distill
  → held-out quality suite
  → safety/tool/retrieval checks
  → quantized export and CPU benchmark
  → register immutable model artifact
```

### Harness lifecycle

```text
request → policy → context → model → action schema → approval → tool → observation → trace
```

Required harness components are a typed tool protocol, capability policy, approval provider,
bounded executor, memory adapter, model adapter, step/token/time budgets, cancellation token,
redacted event trace, replay runner, and regression suite. The first safe implementation should
support deterministic local Python tools only; subprocess/network/filesystem tools come later and
must be explicit capabilities.

## Delivery sequence

This is the order that minimizes rework and gives a useful product after each slice.

### Phase 0 — foundation and truthfulness (completed)

- Restore package/docs/license/ignore files and keep frontend/Rust lockfiles visible.
- Record the actual architecture and limitations in this document.
- Add a canonical agent instruction ledger and a single local test command.
- Remove generated caches and stale starter artifacts only when they have no runtime caller.
- Make docs stop claiming unsupported model backends are complete.

**Exit evidence:** clean checkout installs, Python tests run, web build/lint are reproducible when
dependencies are installed, native tests are reproducible, and all failures are recorded.

### Phase 1 — artifact and evaluation spine (implemented)

The current Phase 1 slice now provides:

- additive versioned artifact manifests with content fingerprints and atomic writes;
- run manifests for base training, DPO, and distillation provenance;
- checkpoint manifest sidecars without changing checkpoint payload schema 2;
- JSONL task suites with exact-match, structured JSON, tool-call, and lexical grounding checks;
- complete case snapshots, latency, explicit generator errors, aggregate summaries, and report
  files (`report.json`, `cases.jsonl`, `failures.jsonl`);
- `nanoforge evaluate-suite` for Nanoforge and imported model engines.

The phase intentionally does not claim semantic grounding, tool execution, release thresholds, or
general intelligence. Those remain later phases.

### Phase 1A — coding-agent boundary (implemented slice)

The next product direction is coding-focused. The first backend slice now provides bounded
repository inspection, expected-hash patching, a fixed pytest profile, structured tool schemas,
capability/approval checks, budgets, cancellation checks, trace events, dry-run replay, and FIM
formatting for code-specialist data. See `docs/coding-agent-architecture.md`.

This is not yet a frontier coding model, imported-model adapter, or hardened OS sandbox. The next
coding slices are repository snapshot/dependency packing, coding-case evaluation, model-specific
FIM/chat/action adapters, and an explicit process/container sandbox.

### Phase 2 — CPU quality path

Add model presets based on measured parameter/activation budgets; fix effective-batch semantics;
implement token-budget batching and optional prefetch; add CPU thread controls and a benchmark
report; add a native or trusted external quantized inference path; add adapter training for an
imported base model; add teacher-to-student distillation with a real teacher checkpoint.

**Exit evidence:** a fixed CPU budget experiment compares scratch, adapter, retrieval, and
distilled variants on the same task suite and reports quality/latency/memory trade-offs.

### Phase 3 — safe agent harness

Replace the minimal direct callable registry with typed schemas and explicit `ToolSpec` fields:
capabilities, input/output schema, timeout, output limit, approval policy, side-effect class,
and audit name. Add a bounded state machine, cancellation, structured action parser, redacted
trace, replay, and deterministic local-tool tests. Make the server and web UI use the same service
boundary instead of their own ad hoc subprocess behavior.

**Exit evidence:** valid calls execute, malformed/unknown/unauthorized/over-budget calls do not;
the trace replays to the same result; no tool runs without the declared approval policy.

### Phase 4 — memory, retrieval, and product workflows

Add governed memory stores with scopes (`session`, `project`, `user`, `global`), owners,
provenance, retention, deletion, consent, and retrieval citations. Add data connectors, dataset
versioning, project manifests, background jobs, resumable uploads, and web views for artifacts,
evaluations, traces, and approvals.

**Exit evidence:** a user can import/adapt/evaluate/register/run an agent locally without copying
private data into source or losing provenance.

### Phase 5 — deployment and scale

Add authenticated non-local serving, request limits, multi-user authorization, model lifecycle
management, continuous batching where justified, real paged KV cache only if benchmarks support
it, stronger distributed checkpointing, and native kernels. Keep these after the local single-user
workflow is correct.

## Acceptance criteria for a “good local agent model”

Nanoforge must not call a model good because its training loss fell. A release candidate needs:

- held-out loss/perplexity and task-level scores;
- structured output/tool-call validity and invalid-call refusal rate;
- retrieval answer accuracy and citation/grounding checks;
- multi-turn state consistency and context-overflow behavior;
- latency (time to first token and tokens/sec), peak RSS/VRAM, and model artifact size;
- deterministic replay with recorded seed/config/artifact fingerprints;
- failure examples, not only averages;
- an explicit statement of whether the model was trained from scratch, adapted from a base, or
  distilled from a teacher.

## Cleanup policy and audit result

### Safe to remove as generated output

- Python `__pycache__/` trees and `.pyc` files.
- `.pytest_cache/`, `.ruff_cache/`, and `src/nanoforge.egg-info/`.
- `native/nanoforge-tokenizers/target/`.
- Generated tokenizer reports under ignored `reports/` locations.
- Unreferenced Vite starter assets `web/src/assets/react.svg`, `vite.svg`, and `hero.png`.

### Keep out of the reusable source contract

- `data/persona.jsonl`, `data/persona_index/`, private chat exports, private RAG indexes, model
  registries, and `scripts/*nashwa*.py`. These are local/private artifacts, not framework source.
  They must not be deleted automatically because they may be the user's only copy.
- `data/tokenizers/`, `data/packed/`, `runs/`, `models/`, and `reports/` remain ignored runtime
  locations and should be regenerated or backed up outside the repository when needed.

### Keep and track

- Python package metadata, README, license, contributing guide, changelog, `.python-version`,
  Docker/Make workflow files, and corrected `.gitignore`.
- `web/package.json` and `web/package-lock.json`.
- Native `Cargo.lock` and Rust source.
- Configs, docs, source, tests, CI, and only documented portable scripts.

## Sources and research notes

Primary sources consulted for this audit:

- Kaplan et al., **Scaling Laws for Neural Language Models** —
  <https://arxiv.org/abs/2001.08361>
- Hoffmann et al., **Training Compute-Optimal Large Language Models (Chinchilla)** —
  <https://arxiv.org/abs/2203.15556>
- Eldan and Li, **TinyStories** — <https://arxiv.org/abs/2305.07759>
- Liu et al., **MobileLLM** — <https://arxiv.org/abs/2402.14905>
- Hu et al., **LoRA** — <https://arxiv.org/abs/2106.09685>
- Dettmers et al., **QLoRA** — <https://arxiv.org/abs/2305.14314>
- Zhou et al., **LIMA: Less Is More for Alignment** — <https://arxiv.org/abs/2305.11206>
- Hinton et al., **Distilling the Knowledge in a Neural Network** —
  <https://arxiv.org/abs/1503.02531>
- ggml-org, **llama.cpp** — <https://github.com/ggml-org/llama.cpp>
- PyTorch, **Performance Tuning Guide** —
  <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>

These sources support design decisions; they do not prove that Nanoforge implements their
algorithms. Every future claim must be tied to a local test, benchmark, or artifact.

## Coding-agent direction (current next slice)

The product direction is now explicitly coding-agent focused. See
[`coding-agent-architecture.md`](coding-agent-architecture.md) for the detailed design and
research notes.

The repository now contains a backend-only bounded agent interface under `src/nanoforge/agent/`:

- repository-relative list/read/search tools;
- expected-hash atomic patching;
- one fixed, allowlisted pytest profile;
- schema validation and duplicate-safe tool registration;
- capability and approval checks;
- step/tool-call/wall-clock budgets;
- cancellation checks;
- bounded outputs and trace events;
- dry-run replay.

This is deliberately not presented as a frontier model or hardened OS sandbox. The next work is
to add coding-case evaluation, imported-model action adapters, FIM-aware tokenizer validation,
repository snapshot/dependency packing, and then an explicit OS/container execution boundary.
