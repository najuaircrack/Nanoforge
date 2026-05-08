# nanoforge-tokenizers

Optional Rust tokenizer acceleration for Nanoforge.

This crate exposes a Python module named `nanoforge_tokenizers`. Nanoforge discovers it at
runtime and falls back to pure Python when it is not installed.

## Build For Local Development

```powershell
pip install maturin
cd native\nanoforge-tokenizers
maturin develop --release
```

Then verify:

```powershell
nanoforge tokenizer-status
nanoforge benchmark-tokenizer --input data/raw --tokenizer byte-native
```

## Current Scope

- Native byte tokenizer with Nanoforge-compatible token IDs.
- Parallel batch byte tokenization through Rayon.
- Native byte-level BPE tokenizer/trainer compatible with Nanoforge's JSON BPE artifacts.
- UTF-8 recovery helper for future ingestion integration.

## Planned Scope

- WordPiece and Unigram LM tokenizers.
- Streaming mmap corpus readers.
- SIMD UTF-8 validation and normalization.
- Rust parquet/Arrow ingestion adapter shared with the Python data pipeline.
