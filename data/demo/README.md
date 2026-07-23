# data/demo/chat.jsonl

A small reproducible subset of [UltraChat](https://huggingface.co/datasets/stingning/ultrachat)
used to demonstrate Nanoforge's structure-aware dataset detection and demo model training.

## Schema

Each line is a JSON object with the original UltraChat shape:

```json
{"id": "150267", "data": ["user turn", "assistant turn", ...]}
```

`data` is a strictly alternating list: even indices are user turns, odd indices are assistant
turns. Nanoforge's profiler detects this automatically and applies `assistant_only` loss masking.

## Regenerate

```bash
python scripts/make_demo_dataset.py --n 800 --out data/demo/chat.jsonl
```

For the larger training set (not committed):

```bash
python scripts/make_demo_dataset.py --n 30000 --out data/demo/train30k.jsonl
```

## Source

Subset of `data/raw/train_1.jsonl` (UltraChat `train_1`, ~149k conversations).
First 800 records by line order — deterministic, no RNG.
