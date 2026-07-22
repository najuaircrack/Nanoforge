# Nano chat model — quickstart

A tiny (~3.5M param) byte-level chat model trained from scratch with Nanoforge.
It greets you, remembers the user's name (Najwan), answers a handful of basic
facts, and says **"I'm sorry, I don't know that."** for anything outside its
training set.

## Environment

```bash
# One-time setup (Windows, Git Bash). Torch CPU wheels come from PyTorch's index;
# the rest from PyPI.
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install pyyaml tqdm tokenizers
.venv/Scripts/python.exe -m pip install -e . --no-deps
```

## Rebuild the model from scratch

```bash
# 1. Generate the synthetic chat corpus (data/raw/chat.jsonl)
python scripts/make_chat_data.py

# 2. Pack it into token/label memmaps (byte tokenizer, assistant-only loss)
.venv/Scripts/python.exe -m nanoforge.cli prepare \
  --input data/raw/chat.jsonl --out data/packed/nano-chat \
  --tokenizer byte --mode chat --loss-masking assistant_only \
  --seq-len 256 --val-fraction 0.02

# 3. Train (~20 min on CPU, 1200 steps)
.venv/Scripts/python.exe -m nanoforge.cli train --config configs/nano-chat.yaml
```

## Chat with it

```bash
.venv/Scripts/python.exe -m nanoforge.cli chat --checkpoint runs/nano-chat/best.pt
```

The `chat` command defaults to chat sampling mode and injects the same short
system prompt the model was trained on. Override the prompt with `--system "..."`
or force exact recall with `--deterministic`.

## Design notes (why it's built this way)

- **Byte tokenizer (vocab 260).** No tokenizer training, deterministic, and it
  reproduces exact strings — ideal for a tiny memorization model.
- **Canonical answers.** Each user intent maps to exactly ONE assistant reply;
  variety lives only on the user side (many phrasings -> one answer). A clean
  function is what makes greedy decoding reproduce answers crisply. Paraphrasing
  the assistant side creates a many-to-many mapping that derails greedy decode.
- **Short system prompt.** Nanoforge's boundary packer truncates each training
  prefix to its last ~seq_len/2 tokens (see `data/modes._encode_role_boundary_sequences`).
  A long system prompt would be front-truncated in training but sent whole at
  inference, so the assistant marker lands at an unseen position and the model
  emits EOS immediately (empty replies). Keep the system prompt short so training
  and inference see identical structure.
