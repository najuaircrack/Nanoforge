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

CPU-only training is supported for tiny models and data-pipeline debugging. Ryzen CPUs benefit
from packed contiguous token buffers and larger batch prefetching. GPU training benefits from
pinning, persistent workers, and packed fixed-length sequences.

## Model quality strategy

Small models need excellent data more than exotic architecture. For code:

- deduplicate aggressively
- keep repository boundaries and file paths as metadata tokens when possible
- mix docs, tests, examples, issues, and explanations
- avoid large volumes of broken generated code
- curriculum from short clean files to long multi-file contexts
- use validation sets by repository, not just random chunks

