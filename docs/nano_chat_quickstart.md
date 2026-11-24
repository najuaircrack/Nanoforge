# Public nano-chat smoke experiment

This is a tiny byte-tokenizer pipeline experiment using twelve newly authored CC0 conversations.
It has no claim of broad knowledge, dependable refusal behavior, personal memory or general assistant quality.

Create a virtual environment and install the package:

```text
python -m pip install -e ".[dev,tokenizers]"
python scripts/make_public_fixtures.py --kind chat --out data/demo/public-chat
python -m nanoforge prepare --input data/demo/public-chat --out data/packed/nano-chat-smoke --tokenizer byte --mode chat --loss-masking assistant_only --seq-len 128 --val-fraction 0.25 --no-progress
python -m nanoforge train --config configs/nano-chat-smoke.yaml --no-tui
python -m nanoforge evaluate --checkpoint runs/nano-chat-smoke/last.pt --data data/packed/nano-chat-smoke/val.bin --seq-len 128 --batches 5 --device cpu
python -m nanoforge chat --checkpoint runs/nano-chat-smoke/last.pt --deterministic
```

The output dataset directory must be new; the fixture writer refuses an existing directory.
The config matches byte vocabulary 260, sequence length 128 and the prepared dataset paths.
It has no dependency on an existing checkpoint.

The shortest verified public path is `python scripts/run_smoke.py --out runs/public-smoke`,
which creates its own data/config and executes a small CPU train/save/evaluate cycle.
See [training.md](training.md) for real-data safeguards, imported-base adaptation and evaluation limits.
