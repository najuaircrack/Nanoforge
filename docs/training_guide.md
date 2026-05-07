# Training Guide

## Implementation plan

1. Build a tokenizer on your target code and text mix.
2. Pack data into `train.bin` and `val.bin` with document-level deduplication.
3. Run `nanoforge params` and check the model fits your VRAM budget.
4. Train a tiny model first to validate loss, checkpointing, and sampling.
5. Scale width before depth for very small models; scale depth once width is at least 512.
6. Add longer context only after short-context loss is stable.
7. Fine-tune with LoRA on instruction and debugging conversations.
8. Distill from a larger teacher if you have high-quality prompts and completions.
9. Export for inference after validation loss and qualitative code samples improve together.

## Dataset recommendations

Use a deliberate blend:

- 45 percent source code from Python, C/C++, JavaScript/TypeScript, Rust, and Pawn/SA-MP
- 20 percent tests, docs, examples, READMEs, and API references
- 15 percent high-quality coding conversations and debugging traces
- 10 percent math, logic, and structured reasoning text
- 10 percent general clean prose for language fluency

For code models, keep licensing metadata. Validate by repository split to avoid measuring
memorization of near-duplicate files.

## Curriculum

Start with short, syntactically clean files and examples. Gradually increase:

- max sequence length
- multi-file snippets
- incomplete autocomplete examples
- bug-fix and refactor prompts
- tool-call or function-call traces

## Hardware recommendations

- CPU only: 10M model, seq_len 256-512, batch 1-4, use byte tokenizer for debugging.
- 4 GB GPU: 10M-35M, BF16/FP16, gradient checkpointing, micro_batch_size 1-2.
- 8 GB GPU: 35M-110M, seq_len 2048-4096, GQA, gradient accumulation.
- 12 GB GPU: 110M with 4096 context comfortably, 250M with careful checkpointing.
- 24 GB GPU: 250M-500M, longer context, LoRA fine-tuning, distillation.

Ryzen CPU tips:

- use memmap packed data
- keep workers modest to avoid RAM pressure
- prefer larger contiguous batches when enough memory exists
- set PyTorch thread count explicitly during benchmarking

## Optimization strategies

- GQA reduces KV cache memory from `n_heads` to `n_kv_heads`.
- Tied embeddings save parameters and usually improve small-model quality.
- RMSNorm avoids mean-centering and is cheaper than LayerNorm.
- SwiGLU improves quality per parameter over plain GELU MLPs.
- Flash SDPA selects efficient kernels when the backend supports them.
- Gradient accumulation simulates larger batches without increasing activation memory.
- Checkpointing trades compute for memory and is essential on low VRAM.
- EMA can slightly improve evaluation stability for noisy small runs.
- LoRA makes fine-tuning cheap and prevents full-model overfitting on small instruction data.

## Scaling guidance

Approximate recipes:

- 10M: `d_model=256`, `layers=6`, `heads=4`, `kv_heads=2`
- 35M: `d_model=512`, `layers=8`, `heads=8`, `kv_heads=2`
- 110M: `d_model=768`, `layers=12`, `heads=12`, `kv_heads=4`
- 250M: `d_model=1024`, `layers=18`, `heads=16`, `kv_heads=4`
- 500M: `d_model=1280`, `layers=24`, `heads=20`, `kv_heads=5`

Train with at least 20 tokens per non-embedding parameter for initial competence. Code-focused
models benefit from more repeated exposure to curated high-signal examples than from simply
maximizing raw token count.

