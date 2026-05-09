# Automatic Training Guide

`nanoforge auto-train` runs the full local workflow:

1. inspect the dataset,
2. infer or apply the training mode,
3. train the selected tokenizer,
4. prepare packed data with the correct masking strategy,
5. write a CPU-friendly config,
6. start training unless `--no-train` is passed.

## Chat Model From UltraChat/ShareGPT

```powershell
nanoforge auto-train ^
  --input data/raw ^
  --name ultrachat-18m ^
  --mode chat ^
  --tokenizer native-bpe ^
  --vocab-size 8000 ^
  --text-column messages ^
  --seq-len 512
```

This writes:

- tokenizer: `data/tokenizers/ultrachat-18m-bpe.json`
- packed data: `data/packed/ultrachat-18m`
- config: `configs/ultrachat-18m.yaml`
- run: `runs/ultrachat-18m`

For a dry setup without training:

```powershell
nanoforge auto-train ^
  --input data/raw ^
  --name ultrachat-18m ^
  --mode chat ^
  --tokenizer native-bpe ^
  --vocab-size 8000 ^
  --text-column messages ^
  --seq-len 512 ^
  --no-train
```

## Generative Model

```powershell
nanoforge auto-train ^
  --input data/raw ^
  --name stories-18m ^
  --mode generative ^
  --loss-masking none ^
  --tokenizer native-bpe ^
  --vocab-size 8000 ^
  --seq-len 1024
```

## Instruct Model

```powershell
nanoforge auto-train ^
  --input data/alpaca.jsonl ^
  --name instruct-18m ^
  --mode instruct ^
  --loss-masking completion_only ^
  --tokenizer native-bpe ^
  --vocab-size 8000 ^
  --seq-len 512
```

## Auto Mode

`--mode auto` inspects dataset fields:

- `messages` or `conversations` -> `chat`
- `instruction` with `output` or `response` -> `instruct`
- `code` -> `code`
- otherwise -> `generative`

For chat data, `auto-train` also defaults `--text-column messages` when the dataset exposes that field.

## Data Quality Check

After preparation, check label masks:

```powershell
python -c "import numpy as np; labels=np.fromfile('data/packed/ultrachat-18m/train.labels.bin', dtype=np.int32); masked=(labels==-100).sum(); total=len(labels); print(f'Masked: {masked/total*100:.1f}%'); print(f'Unmasked: {(total-masked)/total*100:.1f}%')"
```

Healthy chat/instruct data usually lands around 40-70% masked and 30-60% unmasked.

## Continue Later

If you used `--no-train`, start training with:

```powershell
nanoforge train --config configs/ultrachat-18m.yaml
```

Chat afterward:

```powershell
nanoforge chat --checkpoint runs/ultrachat-18m/best.pt --mode chat
```
