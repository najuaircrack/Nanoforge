# Nanoforge Model Training Steps

This guide is the practical end-to-end path for training a Nanoforge language model locally.
Start with the tiny recipe even if your final goal is a larger code model; it catches data,
tokenizer, and hardware issues quickly.

## 1. Create the environment

```powershell
cd C:\Users\USER\Documents\Nanoforge
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[all]"
```

For a low-dependency smoke run, `pip install -e .` is enough if you use the byte tokenizer.

## 2. Collect training data

Use clean, legally usable data. Good code-model mixes include source files, tests, docs,
examples, issue-style debugging traces, and instruction conversations.

Recommended folder shape:

```text
data/
  raw/
    code/
    docs/
    conversations/
  packed/
  tokenizers/
```

Supported input formats include `txt`, `md`, `json`, `jsonl`, `csv`, `tsv`, `yaml`, `xml`,
`sqlite`, `parquet`, `arrow`, `.zip`, `.tar.gz`, source-code repositories, and HTTP text/JSONL
URLs. Optional formats such as Parquet and Arrow require their matching Python packages.

## 3. Inspect the dataset

```powershell
nanoforge inspect-dataset --input data/raw --limit 2000
```

Look for:

- `records` greater than zero
- expected `formats`
- no repeated decode or schema warnings
- reasonable byte volume for your target model

## 4. Train or choose a tokenizer

For first smoke tests:

```powershell
nanoforge prepare --input data/raw --tokenizer byte --out data/packed/tiny
```

For real code training, train byte-level BPE:

```powershell
nanoforge train-tokenizer --input data/raw --type bpe --vocab-size 32000 --out data/tokenizers/code-bpe.json
```

Alternative tokenizers:

```powershell
nanoforge train-tokenizer --input data/raw --type wordpiece --vocab-size 32000 --out data/tokenizers/code-wordpiece.json
nanoforge train-tokenizer --input data/raw --type unigram --vocab-size 32000 --out data/tokenizers/code-unigram
```

## 5. Evaluate tokenizer quality
```powershell
nanoforge tokenizer-report --input data/raw --tokenizer bpe --tokenizer-path data/tokenizers/code-bpe.json --out reports/tokenizer-code.json
```

Healthy signs:

- low `unk_rate`, ideally `0.0` for byte-level BPE
- higher `chars_per_token` without harming code symbols
- no tiny vocabulary collapse where common syntax explodes into too many tokens

## 6. Pack the dataset

```powershell
nanoforge prepare --input data/raw --tokenizer bpe --tokenizer-path data/tokenizers/code-bpe.json --out data/packed/code --code-only --val-fraction 0.01
```

Packing is streaming and sharded. It writes:

- `train.bin` and optional `train.00001.bin` style shards
- `val.bin` and optional validation shards
- `*.manifest.json` files
- `packing.manifest.json` with preprocessing statistics

## 7. Choose a model config

Start with:

```powershell
nanoforge params --config configs/tiny.yaml
```

Scale only after the tiny run trains and samples correctly.

Typical targets:

- 10M: `d_model=256`, `n_layers=6`, `n_heads=4`, `n_kv_heads=2`, context `512-1024`
- 35M: `d_model=512`, `n_layers=8`, `n_heads=8`, `n_kv_heads=2`, context `2048`
- 110M: `d_model=768`, `n_layers=12`, `n_heads=12`, `n_kv_heads=4`, context `4096`
- 250M: `d_model=1024`, `n_layers=18`, `n_heads=16`, `n_kv_heads=4`, context `4096-8192`

## 8. Train

```powershell
nanoforge train --config configs/tiny.yaml
```

With the live dashboard:

```powershell
nanoforge train --config configs/tiny.yaml --dashboard
```

Open an existing run dashboard without starting training:

```powershell
nanoforge dashboard --run runs/tiny
```

The trainer writes `metrics.jsonl` in the run folder. The terminal progress bar shows live
loss, latest validation loss, perplexity, tokens/sec, learning rate, and gradient norm. The web
dashboard graphs loss, throughput, learning rate, and gradient norm from the same file.

For low VRAM:

- keep `micro_batch_size` small
- increase `grad_accum_steps`
- keep `gradient_checkpointing: true`
- use `precision: bf16` on supported GPUs, otherwise `fp16` or `auto`
- keep context shorter until the run is stable

Optimizer options:

- `adamw` is the default and safest
- `lion` can be memory-light and fast
- `adafactor` is useful for low-memory experiments
- `sophiag` is experimental

## 9. Monitor training

Watch:

- training loss should trend down smoothly
- validation loss should not diverge from training loss
- perplexity should decrease
- token samples should become less repetitive over time

TensorBoard:

```powershell
tensorboard --logdir runs
```

## 10. Evaluate a checkpoint

```powershell
nanoforge evaluate --checkpoint runs/tiny/best.pt --data data/packed/tiny/val.bin --seq-len 512 --batches 20
```

Use validation loss, perplexity, token accuracy, and qualitative samples together. A low loss
with poor samples usually means data leakage, repetition, bad tokenization, or too little
instruction-style data.

## 11. Generate samples

```powershell
nanoforge generate --checkpoint runs/tiny/best.pt --prompt "def quicksort(xs):" --max-new-tokens 200
nanoforge chat --checkpoint runs/tiny/best.pt
```

Beam search:

```powershell
nanoforge generate --checkpoint runs/tiny/best.pt --prompt "Write a Python function" --beams 4
```

## 12. Fine-tune for instructions or coding style

Use LoRA for small curated datasets. Set `lora_rank`, `lora_alpha`, and `lora_dropout` in the
model config, then train on instruction or conversation records. Keep examples high quality:
bug fixes, code review comments, tests, refactors, and concise explanations are worth more than
large volumes of noisy generated text.

## 13. Export

ONNX:

```powershell
nanoforge export --checkpoint runs/tiny/best.pt --format onnx --out exports/tiny.onnx
```

GGUF manifest:

```powershell
nanoforge export --checkpoint runs/tiny/best.pt --format gguf --out exports/tiny-gguf-manifest.json
```

## 14. Troubleshooting

Out of memory:

- reduce `micro_batch_size`
- reduce `seq_len`
- enable `gradient_checkpointing`
- use more `grad_accum_steps`
- train a smaller config first

Loss is unstable:

- lower learning rate
- increase warmup
- check for corrupted data
- verify tokenizer report
- reduce model size for debugging

Samples repeat:

- improve deduplication
- add more high-quality diverse data
- use repetition penalty during inference
- check that validation data is not tiny

Code quality is weak:

- add tests, docs, examples, and real repositories
- keep file extensions and language mix balanced
- include debugging and completion-style examples
- train tokenizer on the same language mix as the model

## 15. CPU-friendly tiny model recipe

This path is for a decent CPU such as a modern Ryzen 5/7 or Intel i5/i7 with 16-32 GB RAM. It
will not produce a frontier model, but it can produce a small model that learns dataset style,
syntax, and short completions well enough to validate the whole stack.

Prepare a small, clean dataset:

```powershell
nanoforge inspect-dataset --input data/raw --limit 2000
nanoforge prepare --input data/raw --tokenizer byte --out data/packed/cpu-tiny --val-fraction 0.05 --min-chars 8
```

Train:

```powershell
nanoforge train --config configs/cpu-tiny.yaml --dashboard
```

Recommended CPU settings:

- use byte tokenizer first so tokenizer training is not a blocker
- keep `seq_len` at `128` or `256`
- use `device: cpu` and `precision: fp32`
- keep `micro_batch_size: 1`
- increase `grad_accum_steps` instead of batch size
- disable gradient checkpointing for very small CPU models because recompute can slow training
- use 1-10 MB of clean text for first tests, then scale to hundreds of MB

For a better tiny model after the smoke run:

- train a BPE tokenizer with `vocab_size` between `8000` and `16000`
- set `vocab_size` in the config to match the tokenizer
- use `d_model: 256`, `n_layers: 8`, `n_heads: 8`, `n_kv_heads: 1`
- keep context at `256` until loss is stable
- train for at least `20M-100M` tokens if the CPU can tolerate the time

A reasonable CPU goal is smooth validation loss, low repetition, and short coherent completions
on the domain data. For stronger coding ability, move the same data pipeline to a GPU config
once the CPU run proves the setup is healthy.
