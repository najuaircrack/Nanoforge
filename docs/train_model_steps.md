# Nanoforge Model Training Steps

This is the short practical checklist for current Nanoforge.

## One-command chat training

```powershell
cd C:\Users\USER\Documents\Nanoforge
.\.venv\Scripts\Activate.ps1

nanoforge auto-train ^
  --input data/raw ^
  --name ultrachat-18m ^
  --mode chat ^
  --tokenizer native-bpe ^
  --vocab-size 8000 ^
  --text-column messages ^
  --seq-len 512
```

This trains the tokenizer, prepares packed chat data, writes `configs/ultrachat-18m.yaml`, and
starts training.

## Prepare only

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

Then:

```powershell
nanoforge train --config configs/ultrachat-18m.yaml
```

## Manual equivalent

```powershell
nanoforge train-tokenizer --input data/raw --type native-bpe --vocab-size 8000 --text-column messages --out data/tokenizers/ultrachat-18m-bpe.json

nanoforge prepare --input data/raw --tokenizer native-bpe --tokenizer-path data/tokenizers/ultrachat-18m-bpe.json --mode chat --loss-masking assistant_only --text-column messages --seq-len 512 --out data/packed/ultrachat-18m

nanoforge train --config configs/ultrachat-18m.yaml
```

## Check masks

```powershell
python -c "import numpy as np; labels=np.fromfile('data/packed/ultrachat-18m/train.labels.bin', dtype=np.int32); masked=(labels==-100).sum(); total=len(labels); print(f'Masked: {masked/total*100:.1f}%'); print(f'Unmasked: {(total-masked)/total*100:.1f}%')"
```

For chat/instruct models, expect roughly 40-70% masked labels.

## Chat with the model

```powershell
nanoforge chat --checkpoint runs/ultrachat-18m/best.pt --mode chat --temperature 0.75 --top-p 0.92 --repetition-penalty 1.1 --frequency-penalty 0.1 --no-repeat-ngram-size 4
```

## Generate text

```powershell
nanoforge generate --checkpoint runs/stories-18m/best.pt --mode creative --prompt "The old city was quiet"
```

## Import external models

```powershell
nanoforge import --model path\to\model.gguf --name my-gguf
nanoforge chat --model my-gguf --mode chat
```
