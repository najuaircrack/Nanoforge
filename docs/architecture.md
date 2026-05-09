# Architecture

Nanoforge is built around a decoder-only transformer with production-friendly extension
points. The default block is:

1. RMSNorm
2. Grouped Query Attention with RoPE and optional sliding window
3. Residual add with residual scaling
4. RMSNorm
5. SwiGLU feed-forward or MoE feed-forward
6. Residual add

## Why this shape

- Decoder-only models are the simplest and strongest fit for generative coding tasks.
- RMSNorm reduces normalization work and is widely used in modern compact LLMs.
- SwiGLU usually improves quality per parameter over ReLU/GELU MLPs.
- GQA keeps inference KV cache small while preserving most multi-head quality.
- RoPE gives a strong default for long-context code and cached decoding.
- Sliding-window attention can extend practical context on low-memory devices.
- MoE is optional because it improves capacity but complicates training stability and batching.

## Hardware strategy

Consumer GPUs are usually bottlenecked by VRAM. Nanoforge therefore prioritizes:

- BF16/FP16 autocast
- activation checkpointing
- tied input/output embeddings
- GQA KV cache reduction
- memmap data loading
- gradient accumulation instead of huge batches
- optional 8-bit optimizer compatibility hooks
- small, focused evaluation intervals

CPU-only training is supported for tiny models and chat experiments. On 8GB RAM, prefer
`fp32`, `num_workers=0`, `pin_memory=false`, small micro-batches, and boundary-aware fixed
sequences for chat/instruct data.

## Data architecture

Nanoforge treats datasets as record streams. Raw text, source trees, JSONL, CSV/TSV, sqlite,
parquet, Arrow, archives, Hugging Face streaming datasets, and HTTP inputs are normalized into
`DatasetRecord` objects before cleaning, tokenization, or packing. Structured readers inspect
schemas, select text fields, skip malformed rows, recover invalid UTF-8 where possible, and
avoid loading full parquet/Arrow tables into RAM.

Tokenizer training uses the same record stream as packing. This keeps BPE/WordPiece/SentencePiece
fitting aligned with preprocessing behavior and prevents accidental tokenization of structured
container bytes such as parquet binaries.

Chat and instruct packing is boundary-aware: records are encoded into fixed training sequences
that start at role/conversation boundaries. Labels are shifted during packing so the trainer can
consume `train.bin` and `train.labels.bin` directly. Assistant-only masking targets assistant
responses while user/system/padding tokens use `-100`.

## Registry architecture

Nanoforge 2.0 components are resolved through typed registries in `nanoforge.registry`.
Registries support lazy `module:attribute` targets, aliases, versions, metadata, validation,
and external plugin discovery through the `nanoforge.plugins` entry-point group.

Core registries currently cover:

- transformer blocks
- attention backends
- FFN backends
- positional embeddings
- normalization layers
- optimizers
- schedulers
- tokenizers
- samplers
- quantization backends

`ModelConfig`, `TrainConfig`, `DataConfig`, and `InferenceConfig` validate registry-backed keys
on load, so a bad config fails before training starts. The default transformer block still uses
the same parameter layout as earlier Nanoforge checkpoints, but it now instantiates attention,
FFN, and normalization modules from registry keys.

External packages can expose a plugin by adding an entry point:

```toml
[project.entry-points."nanoforge.plugins"]
my_plugin = "my_package.nanoforge_plugin:register"
```

The callable receives `ALL_REGISTRIES` and can register new components without editing core code.

## Native extension strategy

Python remains the orchestration and research layer. Performance-critical tokenizer,
normalization, streaming parser, parquet/Arrow ingestion, and preprocessing code should gain
Rust implementations first because those paths need memory safety, streaming IO, and parallelism.
C++ is reserved for low-level tensor kernels, SIMD-heavy CPU inference, quantized runtime paths,
and cache-sensitive systems. Native modules must be optional: Python fallbacks preserve local
usability when a compiler toolchain is unavailable.

The first native target is `native/nanoforge-tokenizers`, a Rust/PyO3 extension discovered as
`nanoforge_tokenizers`. The Python tokenizer API exposes `byte-native` and `native-bpe`, which
use Rust when installed and otherwise fall back to compatible Python implementations. This keeps
packed datasets and checkpoints compatible while allowing local builds to opt into faster
parallel batch tokenization and native byte-level BPE.

## CLI orchestration

`nanoforge auto-train` is the high-level workflow command. It inspects input data, trains the
selected tokenizer, prepares packed data with the right mode and masking, writes a CPU-friendly
config, and starts training. Lower-level commands remain available for manual workflows:
`train-tokenizer`, `prepare`, `new-config`, `train`, `chat`, `generate`, and `import`.

## Model quality strategy

Small models need excellent data more than exotic architecture. For code:

- deduplicate aggressively
- keep repository boundaries and file paths as metadata tokens when possible
- mix docs, tests, examples, issues, and explanations
- avoid large volumes of broken generated code
- curriculum from short clean files to long multi-file contexts
- use validation sets by repository, not just random chunks
