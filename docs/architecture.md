# Runtime architecture and evidence boundaries

Latticekin's compatibility module is `nanoforge`. Runtime behavior lives in
`src/nanoforge/`; executable tests are the evidence for public/failure paths.
The [charter](project-charter.md) defines the product goal, and the
[roadmap](roadmap.md) separates implementation from remaining validation.

## Data and artifacts

Readers produce `DatasetRecord` objects. The data layer applies configured size/network limits,
format normalization and limited pattern-based governance, records filtering outcomes,
packs token/label shards and assigns whole source groups to train/validation.
A zero validation fraction does not manufacture overlapping held-out data.

Streaming behavior is format-specific. Line/chunk readers and optional mmap paths are bounded;
container-format parsing and BPE training retain limited in-memory structures or refuse an oversized
input. Python-BPE fitting has a configured corpus-byte budget. Reported Python allocation peaks do
not represent native/process peak memory. Native Rust byte/BPE/mmap paths retain Python fallbacks.
SIMD normalization and native Arrow/Parquet extraction remain separate roadmap outcomes.

Saved configs, packing manifests, checkpoint/run metadata and persistent traces have explicit
schemas and integrity/provenance fields. Old artifacts may lack origin or parent identity; callers
must surface that absence. Concurrent multi-file publication and model/runtime compatibility
need their own tests; the word atomic describes individual committed writes, not every workflow.

Checkpoint loading requires PyTorch 2.6 or newer and the safe weights-only unpickler. Legacy
config/NumPy types are explicitly allowlisted; unknown pickle types must fail rather than fall back
to unrestricted deserialization. Never load an untrusted checkpoint with an unsafe pickle fallback.

## Model and training

The compact PyTorch model supports dense Transformer blocks, GQA/MQA, tied embeddings,
normalization/position choices, FFNs, task heads, LoRA and experimental MoE.
LoRA has an explicit adapter-only training path; changing the rank alone is not a pretrained
import workflow. Imported Transformers full-SFT/LoRA is a separate API in
`training/pretrained.py`, with local loading and reviewed prompt/response records.

The base trainer controls accumulation, precision, clipping, checkpoint/resume, CPU threads,
EMA/SWA/curriculum/recovery and optional distributed wrappers. Preset sizes and resource estimates
are not throughput or quality measurements. CPU smoke checks do not certify GPU/FSDP/offload behavior.
A teacher checkpoint is required to attribute distillation to a real teacher; a student-initial
copy cannot establish capability transfer.

Lion/Adafactor have compact implementations. The Sophia-inspired optimizer is an experimental
approximation rather than the published Sophia-G curvature-estimation algorithm.
Mamba/RWKV names denote experimental mixers, not full architectures.
Registry metadata and the actual backend must explain available/fallback behavior.

## Generation and export

The generation layer owns sampling, bounded KV/prefix cache state and streaming. Portable
chunked online-softmax, local-block sparse, periodic-global and independently paged K/V attention
now implement reference algorithms. Beam/hidden-state contrastive/Mirostat1/2, greedy/stochastic
speculation, bounded CFG/JSON-byte grammars and grouped-forward continuous scheduling have focused
CPU cases. These are not optimized native/CUDA kernels or a zero-copy serving engine.
External FlashAttention requires compatible package/device/
dtype; version labels do not imply separately validated v2/v3 implementations.

Dynamic int8 and optional external quantized runtimes are distinct paths. QLoRA, GPTQ, AWQ and
native CPU SIMD are not interchangeable capabilities. ONNX is an inference export with a declared
supported subset. Nanoforge's own GGUF export declares a custom architecture for faithful native
reload; it is not a llama.cpp-compatible model export. Imported external GGUF inference through
llama.cpp is a separate path. Binary validity alone does not establish arbitrary runtime compatibility.
Export tests must compare real runtime logits/generation where interoperability is claimed.

## Retrieval, evaluation and agent loop

RAG combines chunking, embedder identity, stored vectors/documents and retrieval.
Governed memory and model-driven coding evaluation are separate interfaces with lifecycle metadata;
a lexical hit or substring citation does not establish semantic faithfulness.

The safe model-facing path is:

```text
project → bounded context → provider → structured action → schema/policy/approval
        → trusted-tool executor → observation/state → redacted trace → final/replay
```

Repository read/write/test tools are bounded, relative and capability-controlled.
The executor uses trusted registered callables in a separate process for deadlines/cancellation,
with platform-specific process-tree/resource controls. This is not an OS filesystem/network
sandbox: registered code and repository tests must be trusted or run inside a separately configured
container/OS boundary. Dry-run replay does not call tools or reapply patches.

The legacy `tools.call_json` API now requires explicit `legacy.execute` capability and human
approval. Migrate model-facing tools to `ToolSpec`, `ToolRegistry` and `ToolExecutor` so
validation, limits and approval remain authoritative.

## Interfaces and checks

The CLI, Textual console and React studio are public callers. Servers remain localhost by default;
nonlocal sharing has a separate policy and security test requirement. The packaged web bundle can
differ from edited web source until the frontend build publishes it.

Use `python scripts/check.py` after installing the documented Python/Node dependencies.
Focused modernization tests cover each subsystem. Remote CI configuration, skipped optional tests,
synthetic fixtures and CPU-only results do not certify real-model task quality or hardware-specific
performance. [roadmap.md](roadmap.md) retains those acceptance gates.
