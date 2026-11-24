# Modernization roadmap and acceptance ledger

Updated November 2026. The product remains experimental. Implementation, focused synthetic
regressions, full release checks, hardware validation and real-model quality are different states.
The historical [platform audit](platform-audit.md) and [2026-11-11 audit](audit-2026-11-11.md)
are preserved; this table is the current task/disposition index.

The initial useful product is a private coding/workspace agent: a developer selects a licensed
local model, sees its proposed repository change, approves effects, runs a fixed test profile,
and inspects a redacted trace plus held-out task/resource results. Scratch training stays an
educational/research/specialist lane.

## Implemented foundations and current verification boundaries

Each row names an observable check. A test file proves only the cases it exercises; it does not
certify all models, architectures, corpora or hardware. Parent integration results belong in the
project-local fix-progress ledger.

| ID | Outcome and current state | Observable acceptance |
| --- | --- | --- |
| REL1 | Source-only Docker/archives, no implicit runtime copy; source controls implemented. | Release privacy regressions pass; build a synthetic-control image and prove private paths absent. Live Docker gate remains pending where Docker is unavailable. |
| REL2 | Public authored C/chat fixture generator, matching configs, no-overwrite behavior and CPU smoke implemented. | tests/test_release_modernization.py trains/saves/reloads/evaluates only freshly authored data and tests overwrite refusal. |
| REL3 | Package extras and actual source archive/wheel installation implemented. | Release tests build archives offline, inspect contents, install wheel into a new environment and execute the public CLI without source PYTHONPATH. |
| REL4 | Portable offline check and Linux/Windows/optional/web/Rust/package/container CI defined. | python scripts/check.py plus requested flags succeeds after dependencies installed; remote CI jobs must actually run before a release claim. |
| REL5 | Schema/CLI references generated from code; stale-document check implemented. | Both generator --check commands match final schemas/parser; full release suite catches drift. |
| CFG1 | Strict config/duplicate-key/type/finite/range validation, atomic save and effective-batch semantics implemented. | Root config integration cases exercise invalid keys/YAML/types, roundtrip and derived batch values. |
| D1 | Bounded records/sources/container parsing, explicit network and source-group split implemented. | Data regressions reject oversized/malformed/unapproved inputs and show no train/validation group overlap. |
| D2 | Canonical roles, assistant/completion labels and internal-span FIM implemented. | Data/FIM tests verify prompt/response semantics, shifted targets, mask integrity and vocabulary bounds. |
| D3 | Limited pattern PII/license/consent policy and filtering counters implemented. | Synthetic governance cases reject declared patterns and retain counts. This is not comprehensive secret/PII detection. |
| D4 | Lower-memory bounded Python-BPE fitting, optional Rust mmap reader and fallback implemented. | Native/Python tokenizer and reader interoperability tests pass with and without the extension; measured Python allocation peak is labeled separately from native/process RAM. |
| D5 | Bounded repository snapshots/dependency/exclusion helpers implemented. | Synthetic snapshot cases respect byte/entry/path limits. Full language dependency resolution is not claimed. |
| M1 | Checkpoint closure, attention/capping/positions/cache/UTF-8/MoE/sampler correctness repairs implemented. | tests/test_model_modernization.py compares public/cache/failure paths against appropriate CPU references. |
| M2 | Chunked online-softmax, local-block sparse, periodic-global and independent paged-K/V attention implemented as portable references. | CPU tests compare supported semantics with dense references, including paged continuation. Optimized CPU/CUDA speedups require independent benchmarks. |
| M3 | Bounded prefix LRU, beam/hidden-state contrastive/Mirostat1/2, stochastic/greedy speculation and bounded grammar interfaces implemented. | Model tests check algorithms, tokenizer identity, stop/UTF-8/budget/error paths; task quality and real GPU behavior remain separate gates. |
| M4 | Continuous admission/cancellation/deadline/token scheduler with grouped model forwards implemented. | Queue tests exercise concurrent requests, reservations, cancellation and grouped forward equivalence. K/V copying/merging is not zero-copy paged serving. |
| T1 | Resume/RNG/sampler/checkpoint/recovery/accumulation/metric fixes, CPU threads, EMA/SWA/curriculum and offload controls implemented. | Training regression suites compare interrupted and uninterrupted CPU runs, failed updates and public control semantics. |
| T2 | Distributed synchronization/no_sync/consolidation/ZeRO hooks implemented. | CPU distributed tests plus real multi-GPU DDP/FSDP save/resume/offload checks are required; absent hardware remains visible. |
| A1 | Imported Transformers full SFT/LoRA, frozen-base/adapter artifacts and local reload API implemented. | A tiny locally constructed model/tokenizer trains/reloads without downloads and changes adapter outputs while preserving base parameters. |
| A2 | Alignment masking/teacher/tokenizer, export/quantization/import integrity repairs implemented. | Adaptation/export regressions exercise real supported runtime reload/logits comparisons and negative formats. GPU QLoRA and arbitrary external architectures are not certified by CPU checks. |
| AG1 | Schema dialect, policy approval, action/token/context/output limits and process cancellation implemented. | tests/test_agent_modernization.py rejects malformed/unauthorized/oversized calls, terminates trusted workers and preserves policy-required approval. |
| AG2 | Persisted checksummed redacted traces and dry-run control-state replay implemented. | Synthetic traces exclude payload/argument values by default, reject corruption and replay without tool effects. |
| AG3 | Windows Job tree termination/memory limit and POSIX process-group/per-process limits implemented. | Platform tests measure actual settlement/resource failure. This does not impose an OS filesystem/network sandbox. |
| W1 | Schema-2 manual/native/imported project selection, constrained context, interactive approvals and TUI operator paths implemented. | Workspace/TUI tests cover provider selection, stale hashes, paths, ignored/sensitive files, approved/denied effects and cancellation. |
| B1 | Isolated local studio state, transactional bounded upload/config, job/log/operator controls implemented. | HTTP regressions cover traversal, collision, malformed JSON, denied patch, cancellation and persisted dry replay. |
| B2 | OpenAI-style completion/chat/SSE envelopes and measured visible-text usage, per-model serialization/capacity settlement implemented. | HTTP/native CPU tests cover stream/failure/concurrency limits; visible-text retokenization is not exact hidden-token accounting. |
| B3 | Nonlocal bearer/origin/body/token/concurrency/deadline policy implemented. | Security tests reject missing/short keys and unauthorized origins/requests. TLS, tenants and OS isolation remain separate work. |
| FE1 | Studio workflow contracts, operator/artifact/evaluation pages and genuine frontend tests implemented. | Vitest workflow suite, TypeScript/Vite build and lint pass against real API contracts; Python wrapper requires installed Node/web dependencies. |
| EV1 | Versioned loss/task/structured/tool/lexical-grounding reports and failed cases exist; modernization owns metric/index/artifact correctness. | Evaluation regression suites verify masked denominators, output/error cases, index identity, artifact integrity and report thresholds. |

## Required remaining outcomes

These are concrete future gates, not permission to describe the whole roadmap as complete.
Some already have partial interfaces. Close them only with an end-to-end implementation and the
specified evidence; unsupported hardware is an explicit validation gap.

| ID | Required outcome | Smallest decisive acceptance gate |
| --- | --- | --- |
| Q1 | Useful model-driven local coding quality and failure recovery. | Licensed local coding model completes a public held-out repository repair set; inspect approved diffs, tests, denied calls, stale patches, recovery and unchanged unrelated files. |
| Q2 | Fixed CPU-budget comparison of scratch, adapter, retrieval and distilled variants. | Same task suite/source split/seed/hardware reports pass rate, failure cases, TTFT, tokens/sec, peak RSS/VRAM and artifact size within measured budgets. |
| Q3 | Versioned instruction/refusal/safety/multi-turn/context-overflow regression gates. | Each case has a meaningful oracle; thresholds reject a lower-loss but worse-task candidate. |
| Q4 | Retrieval semantic faithfulness and citation validity beyond substring checks. | Held-out answer/source cases include misleading evidence and citation mismatches, with independently reviewed semantic failure examples. |
| MEM1 | Governed session/project/user/global memory end-to-end. | Owner/scope/provenance/consent/retention/deletion enforced across CLI/TUI/studio and persistence; revoked/deleted facts cannot be retrieved or used for training. |
| DATA1 | Complete dataset-quality report and governed dataset versions/connectors. | Report UTF-8 health, duplicates/near duplicates, repetition, entropy, language mix, code/text balance, token lengths and vocabulary diversity with counts/fingerprints and failure controls. |
| DATA2 | Full repository/snapshot and language dependency-aware packing. | Multi-repository fixture preserves snapshot/group boundaries and records unresolved dependencies/exclusions without silent drops. |
| TOK1 | Native WordPiece and Unigram plus richer collapse/merge-quality diagnostics. | Native/Python serialization/encode/decode fixtures and independent quality/memory measurements pass. Registry names alone do not deliver native algorithms. |
| TOK2 | SIMD normalization/UTF-8 and native Arrow/Parquet extraction. | Real compiled implementations match Python behavior on malformed Unicode and batch schemas and measure throughput/memory. Existing mmap/byte/BPE does not satisfy this gate. |
| CPU1 | Reproducible target-hardware performance profiles and RAM-pressure/prefetch tuning. | MKL/OpenMP/Ryzen/thread/worker/batch/offload profiles measure task quality/hour, actual peak memory and cancellation under pressure. |
| HEALTH1 | Deeper training diagnostics and activation partition/offload recovery. | Synthetic faults identify NaN sources, attention entropy, saturation/dead channels, hidden-state statistics and memory fragmentation; activation partition/offload changes compare outputs, peak memory and recovery against the baseline. Existing loss/gradient/logit-entropy checks do not establish these diagnostics. |
| NATIVE1 | Optimized CPU SIMD/weight-only quantized inference hot path. | Native or trusted external runtime beats the agreed baseline without output/task regression; quantify kernels/runtime/tokenizer compatibility. |
| GPU1 | Real CUDA FlashAttention versions/QLoRA/FSDP/offload/distributed quality. | Run device-specific integration/checkpoint/reload/resource experiments. Flash-v3 remains unavailable until a real implementation exists. |
| ALG1 | Full SSM/RWKV or published Sophia-G implementation if retained as a product goal. | Algorithm-specific reference equivalence and benchmark cases; current experimental mixers/gradient-square approximation retain honest metadata. |
| EXP1 | Broader GGUF/ONNX/TensorRT/runtime model/tokenizer compatibility. | Real runtime load/logit/generation equivalence for every declared supported subset; reject unsupported heads/positions/tokenizers. |
| SBX1 | OS/container filesystem/network/credential sandbox for untrusted tests/tools. | Attempted outside-workspace writes/network/credential access are prevented by the actual boundary, not merely application path checks or trusted-process limits. |
| JOB1 | Resumable uploads/background-job lifecycle and artifact/eval integration. | Interrupted upload resumes without duplication/corruption; cancellation/worker death/restart retains consistent manifests and UI-visible failures. Current atomic bounded upload is not resumable. |
| SHARE1 | TLS, multi-user authorization/tenant isolation and controlled sharing. | Separate users cannot access each other's models/data/traces/capabilities; authenticated remote limits and security tests run under the actual deployment. |
| ART1 | Universal immutable model/artifact origin and governed lineage. | Imported/scratch/adapted/distilled/composed artifacts preserve base/teacher/tokenizer/source identities; migration exposes unknown legacy fields; annotated runs retain immutable evidence. |
| ADAPT1 | Exact imported-adaptation resume. | Interrupt/restart SFT/LoRA with optimizer, scheduler, RNG, tokenizer/data identities and sample order; final parameters/metrics match the uninterrupted baseline. Current save/reload alone does not satisfy this gate. |
| REL6 | Live container build and remote CI certification. | Source-only image installs/runs CLI and rejects synthetic private-context controls; every declared Linux/Windows/optional/web/native/package job actually passes. |

## Delivery order

Fix and reverify local correctness/release/privacy first. Close Q1 with one licensed local model,
then Q2/Q3 and memory/retrieval governance. Choose performance/native/hardware work from measured
bottlenecks. Stronger untrusted-tool sandboxing precedes riskier capabilities; multi-user sharing
comes after those boundaries.

Each milestone must update callers, saved schemas, docs and negative tests together. Preserve
failed/unavailable gates and rollback paths. A successful tiny CPU run or a newly present UI page
does not complete the platform.
