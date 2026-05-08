# Nanoforge Roadmap

Nanoforge is moving toward a local-first transformer research stack that stays small enough
to read, but has clear extension points for modern training, inference, and data systems.

## Architecture Registries

- Expand registries for attention backends, FFN variants, positional embeddings, optimizers,
  schedulers, tokenizers, sampling algorithms, normalization layers, and quantization backends.
- Keep checkpoint keys stable while allowing config-only swaps between implementations.
- Add plugin discovery for external research modules without changing core imports.
- Phase 0 status: core registries, lazy factories, plugin discovery, config validation, and
  registry-backed default transformer assembly are implemented. Remaining work is to migrate
  every experimental future backend from placeholder metadata to real implementations.

## Structured Data Ingestion

- Treat structured formats as schemas, not byte blobs.
- Stream parquet and Arrow with chunked readers, automatic text-column detection, configurable
  column selection, corrupt-row skipping, invalid UTF-8 recovery, schema previews, field stats,
  dataset fingerprints, and duplicate-dataset detection.
- Maintain dedicated adapters for parquet, Arrow, Hugging Face streaming datasets, sqlite,
  csv/tsv, JSON conversations, OpenAI chat, ShareGPT, instruction datasets, code repositories,
  archives, and HTTP streams.

## Tokenizer Stability

- Train tokenizers from streaming text iterators instead of materializing full corpora.
- Validate datasets before fitting, recover from malformed UTF-8, skip bad rows, expose dry-run
  reports, memory profiles, unknown-token diagnostics, token distribution summaries, vocabulary
  collapse checks, and merge-quality reports.
- Add native tokenizer engines in Rust first, with Python orchestration and graceful fallback to
  pure Python or `tokenizers`/SentencePiece adapters when native extensions are unavailable.

## Dataset Inspection And Cleaning

- Provide CLI tools for `inspect-dataset`, `validate-dataset`, `clean-dataset`,
  `deduplicate-dataset`, and `convert-dataset`.
- Add quality reports covering UTF-8 health, duplicate and near-duplicate detection, repetition,
  entropy, language mix, code/text balance, token-length histograms, and vocabulary diversity.
- Export cleaned txt, JSONL, parquet, and packed binary shards through reproducible manifests.

## Training Health

- Track gradient explosions, NaN source hints, optimizer instability, attention entropy,
  token entropy, activation saturation, dead channels, hidden-state statistics, memory
  fragmentation, and recommended config fixes.
- Add robust resume, checkpoint schema versions, integrity validation, async saves, anomaly
  recovery, EMA/SWA evaluation, curriculum hooks, DDP/FSDP/ZeRO hooks, CPU offload, and
  activation checkpoint partitioning.

## Inference Engine

- Implement paged KV cache, paged attention, prefix/prompt caching, continuous batching,
  beam search, contrastive search, Mirostat, speculative decoding, grammar-constrained decoding,
  and low-latency streaming.
- Keep export compatibility for GGUF, ONNX Runtime, TensorRT paths, and llama.cpp metadata.

## CPU-First Performance

- Prefer chunked and iterable pipelines, mmap-heavy workflows, adaptive batching, async prefetch,
  cache-aware batching, pinned-memory options, RAM-pressure monitoring, dynamic worker scaling,
  and low-memory fallback modes.
- Add CPU tuning profiles for MKL/OpenMP and Ryzen-oriented presets.

## Native Performance Layer

- Use Rust for tokenizer pipelines, UTF-8 validation, streaming parsers, Arrow/parquet ingestion,
  and safe concurrent preprocessing.
- Use C++ for low-level CPU kernels, quantized inference paths, SIMD-heavy compute, and cache
  systems.
- Keep Python focused on configuration, orchestration, experimentation, and research APIs.
- Current implementation target: `native/nanoforge-tokenizers` provides the first optional
  Rust/PyO3 tokenizer module. `byte-native` auto-detects it and falls back to Python when absent.
- Next native milestones: BPE training/merging, mmap corpus readers, SIMD normalization, and a
  Rust parquet/Arrow text extractor shared by tokenizer fitting and dataset packing.

## Finalization Criteria

- Every major subsystem should include validation, logging, diagnostics, profiling hooks,
  regression tests, and recovery paths.
- The framework should avoid loading massive datasets into RAM, full-corpus tokenizer fitting,
  unnecessary tensor duplication, and blocking preprocessing pipelines.
- Documentation should include architecture diagrams, implementation notes, and practical local
  recipes for low-RAM CPU and low-VRAM GPU systems.
