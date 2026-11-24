# Public C/FIM smoke experiment

The checked-in `scripts/make_public_fixtures.py` authors twelve small C functions under CC0.
It reads no user dataset and refuses to overwrite an existing output directory.
Use the generated directory, not any private files already present in `data/`.

```text
python scripts/make_public_fixtures.py --kind c --out data/demo/public-c-tiny
python -m nanoforge validate-config --config configs/c-tiny.yaml
python -m nanoforge prepare --input data/demo/public-c-tiny --tokenizer byte --mode fim --seq-len 128 --val-fraction 0.25 --out data/packed/c-tiny --no-progress
python -m nanoforge train --config configs/c-tiny.yaml --no-tui
python -m nanoforge evaluate --checkpoint runs/c-tiny/last.pt --data data/packed/c-tiny/val.bin --seq-len 128 --batches 5 --device cpu
```

The distinct source/group identifiers support a held-out split. FIM uses an internal missing span:
`<|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>middle`.
The byte tokenizer does not turn these strings into dedicated special IDs. This is a formatting,
checkpointing and evaluation smoke experiment, not a real code-specialist quality benchmark.

For a fast complete check run `python scripts/run_smoke.py --out runs/public-smoke`.
A real specialist needs a suitable code tokenizer, larger licensed/curated data, repository/snapshot
splits, syntax and execution cases, resource measurements and an explicit task limitation.
