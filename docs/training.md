# Training guide

The quickest reproducible check is `python scripts/run_smoke.py --out runs/public-smoke`.
It authors public CC0 C examples, packs distinct held-out source groups, trains a tiny CPU model,
saves a checkpoint and evaluates it. The output directory must be new. This proves pipeline
correctness; it does not demonstrate useful general coding ability.

## Install and check

Use Python 3.10–3.12. After creating and activating a virtual environment:

```text
python -m pip install -e ".[dev,tokenizers,serve,tui]"
python scripts/run_smoke.py --out runs/public-smoke
```

For the full offline repository check, install Node 24 and run `npm ci` in `web/` first,
then `python scripts/check.py` (or `make test`). Offline means checks do not fetch models
or install dependencies. Optional paths need the extras listed in the README.

Examples below use single-line commands that work in PowerShell, cmd.exe and Unix shells.
Use PowerShell backticks only when splitting a command across lines; cmd.exe's caret syntax
does not work as PowerShell continuation.

## Public C/FIM experiment

These commands use matching checked-in settings and a newly authored dataset:

```text
python scripts/make_public_fixtures.py --kind c --out data/demo/public-c-tiny
python -m nanoforge validate-config --config configs/c-tiny.yaml
python -m nanoforge inspect-dataset --input data/demo/public-c-tiny
python -m nanoforge prepare --input data/demo/public-c-tiny --tokenizer byte --mode fim --seq-len 128 --val-fraction 0.25 --out data/packed/c-tiny --no-progress
python -m nanoforge train --config configs/c-tiny.yaml --no-tui
python -m nanoforge evaluate --checkpoint runs/c-tiny/last.pt --data data/packed/c-tiny/val.bin --seq-len 128 --batches 5 --device cpu
```

The fixture has twelve distinct source groups. FIM removes an internal span and formats
prefix, suffix and the missing middle with the FIM sentinels. Byte tokenization encodes those
sentinels as bytes and is intended only for this smoke experiment. Real coding training should
validate dedicated FIM token IDs and split by repository/snapshot.

## Public chat experiment

```text
python scripts/make_public_fixtures.py --kind chat --out data/demo/public-chat
python -m nanoforge prepare --input data/demo/public-chat --tokenizer byte --mode chat --loss-masking assistant_only --seq-len 128 --val-fraction 0.25 --out data/packed/nano-chat-smoke --no-progress
python -m nanoforge train --config configs/nano-chat-smoke.yaml --no-tui
python -m nanoforge chat --checkpoint runs/nano-chat-smoke/last.pt --deterministic
```

These twelve authored conversations test formatting and supervision. They do not establish
reliable factual answers, memorization, refusal behavior or general knowledge.

## Train a local specialist on your own reviewed data

Inspect, validate and profile before adapting or training. Check filtering counts, tokenizer
identity, label balance and held-out source groups. Never use private runtime data as a test fixture.

```text
python -m nanoforge inspect-dataset --input data/raw
python -m nanoforge validate-dataset --input data/raw
python -m nanoforge profile-dataset --input data/raw
python -m nanoforge auto-train --input data/raw --name local-specialist --mode auto --tokenizer native-bpe --vocab-size 8000 --seq-len 512 --ram 8GB --no-train
python -m nanoforge train --config configs/local-specialist.yaml --no-tui
python -m nanoforge evaluate --checkpoint runs/local-specialist/last.pt --data data/packed/local-specialist/val.bin --seq-len 512 --batches 20 --device cpu
```

`auto-train --no-train` creates a config matching its tokenizer and packed outputs.
The larger `nano-debug`, `small-base`, `chat-sft` and `moe-large` configs are templates:
replace the explicitly named dataset/tokenizer paths before use. Their hardware comments are
estimates, not measured promises.

## Imported-base adaptation

Install `.[pretrained]` (equivalently `.[adaptation]`). The
`nanoforge.training.pretrained.PretrainedAdaptationConfig` and `train_pretrained` API
support bounded full SFT or LoRA from a local Transformers-compatible model directory.
Data records require prompt/response strings; preserve the imported tokenizer and compare
against an explicit held-out set. Network loading is opt-in. See CLI help for available
adaptation commands and [configuration.md](configuration.md) for runtime controls.

After supplying a compatible local Transformers model directory at the template's source path:

```text
python -m pip install -e ".[pretrained]"
python scripts/make_public_fixtures.py --kind sft --out data/demo/public-sft
python -m nanoforge adapt --config configs/pretrained-lora-example.json
```

The authored prompt/response sets are disjoint. This short example verifies an adaptation workflow;
it does not measure useful coding quality. Check the source model license and tokenizer and replace
the small fixture with reviewed task data for real adaptation.

Current pretrained SFT/LoRA artifacts support local save/reload, but exact optimizer/RNG resume is
a separate remaining gate. The adaptation output directory must be empty. Nanoforge's own GGUF
exports use a custom native-reloadable architecture, rather than claiming llama.cpp model compatibility.

QLoRA also needs `.[qlora]`, compatible bitsandbytes hardware and a real GPU validation run.
Installing the extra is not evidence that a particular machine or imported model is supported.
Keep changing/private facts in retrieval instead of automatically retraining on them.

## Packing, resources and reproducibility

Chat/instruct/completion examples use canonical role boundaries. Labels select assistant or
completion targets; inspect actual masking counts rather than treating a percentage heuristic
as a quality guarantee. Long examples may be truncated or segmented according to the recorded policy.

Validation fraction zero means no held-out dataset is manufactured. A nonzero fraction needs
at least two distinct source groups and must not copy training data into validation.
Reader limits bound individual records/sources and cumulative input; large container formats
may be refused rather than silently read without limits. Network access is explicit.
Pattern-based PII rejection is a limited filter, not comprehensive PII detection.

Effective batch derives from micro batch, accumulation and distributed world size. CPU thread
and precision controls, EMA/SWA, recovery and checkpointing are described by the generated
schema reference. Profile peak memory and latency on the intended hardware; do not assume
a memory estimate is a measured training budget.

Runs and checkpoint sidecars carry versioned metadata. Inspect available origin/base/teacher,
data/tokenizer and code/environment identities; legacy artifacts may have incomplete provenance.
A lower training loss alone does not establish a better model.

## Evaluate behavior

```text
python -m nanoforge evaluate-suite --suite configs/evals/smoke.jsonl --checkpoint runs/c-tiny/last.pt --out reports/evaluations/c-tiny
```

Exact match, structured JSON and lexical grounding checks are limited oracles.
Tool-call syntax alone does not prove safe execution; evidence substrings alone do not prove
semantic faithfulness. Coding quality needs executable repairs, denied calls, failure recovery,
held-out source groups and resource reports. Failed cases must remain visible.

For monitoring use `latticekin tui --run runs/c-tiny` or localhost-only
`latticekin web --run runs/c-tiny`. Check [roadmap.md](roadmap.md) for unresolved release gates.
