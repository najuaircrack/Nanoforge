# Latticekin

Latticekin is an experimental local toolkit for connecting a model, reviewed data, retrieval,
structured tools, policy and evaluation. The Python module and compatibility CLI remain
`nanoforge`. The product goal is a private coding/workspace agent whose proposed changes can
be approved, tested and traced on an ordinary workstation.

A tiny scratch model is useful for education or a narrow specialist. It cannot acquire frontier
knowledge, reasoning or reliable tool behavior from a small personal corpus. Prefer a licensed
capable local base, retrieval for changing facts, and curated adaptation when evaluation shows a gap.

## Current implementation

- Configurable PyTorch model/training, GQA/MQA, RoPE, masked data packing, checkpointing and metrics.
- Bounded readers, source-group splits, tokenizer adapters and optional Rust byte/BPE acceleration.
- Local/imported inference adapters, LoRA/SFT APIs, experimental alignment/export paths and task checks.
- Structured agent actions, capability/approval policies, trusted-callable process isolation,
  repository tools, hash-checked patches and persistent redacted traces with dry-run replay.
- Textual training/operator views and a React local studio.

These are implementation paths with focused synthetic checks, not a completed platform or a
guarantee of real-model coding quality. [The roadmap](docs/roadmap.md) records outstanding gates;
[the charter](docs/project-charter.md) defines the intended product. The historical
[2026-11-11 audit](docs/audit-2026-11-11.md) describes the worktree before these repairs.

## Install

Use Python 3.10–3.12 and PyTorch 2.6 or newer; `.python-version` records the default.
Create and activate a virtual environment, then install the capabilities you need:

```text
python -m venv .venv
python -m pip install -e ".[dev,tokenizers,serve,tui]"
```

Activation: `.venv\Scripts\Activate.ps1` in PowerShell,
`source .venv/bin/activate` in a Unix shell.

| Extra | Capability |
| --- | --- |
| `tokenizers` | Python BPE/WordPiece/SentencePiece libraries |
| `data` | Arrow/Parquet and additional dataset readers |
| `serve` | FastAPI/Uvicorn, uploads and system metrics |
| `tui` | Textual terminal interface |
| `pretrained` or `adaptation` | Transformers/PEFT/Accelerate local SFT and LoRA |
| `qlora` | bitsandbytes dependency; needs compatible hardware and separate validation |
| `gguf` | llama-cpp-python external inference runtime |
| `export` | ONNX/ONNX Runtime/ONNXScript |
| `embeddings` | Sentence-transformers retrieval embeddings |
| `native` | Maturin build tooling; build/install the Rust extension separately |
| `dev` | Python tests, lint and packaging checks |

The `all` extra collects the regular optional Python capabilities; heavy native GGUF and
hardware-specific QLoRA remain explicit choices. It does not download a model or build the Rust extension.

## First reproducible workflow

Run a complete authored-data CPU train/save/held-out-evaluate smoke test:

```text
python scripts/run_smoke.py --out runs/public-smoke
```

It refuses to reuse an existing output directory and reads no user dataset. The generated records
are newly authored public CC0 examples with distinct split groups. This verifies pipeline behavior,
not general coding ability or assistant quality.

For the longer public C/FIM experiment:

```text
python scripts/make_public_fixtures.py --kind c --out data/demo/public-c-tiny
python -m nanoforge validate-config --config configs/c-tiny.yaml
python -m nanoforge prepare --input data/demo/public-c-tiny --tokenizer byte --mode fim --seq-len 128 --val-fraction 0.25 --out data/packed/c-tiny --no-progress
python -m nanoforge train --config configs/c-tiny.yaml --no-tui
python -m nanoforge evaluate --checkpoint runs/c-tiny/last.pt --data data/packed/c-tiny/val.bin --seq-len 128 --batches 5 --device cpu
```

See [training.md](docs/training.md) for matching chat, real-data and imported-adaptation workflows.
The fixture generator is shipped in the source archive; no personal `data/` file is required.

## Local coding/workspace console

```text
latticekin tui --project configs/project-example.yaml
latticekin tui --run runs/c-tiny
```

The checked-in project grants read-only inspection and scopes itself to its manifest directory.
Copy it into the root of the intended repository. Project schema 2 can select manual, native
checkpoint or imported-model providers and can request interactive write/test approval.

A local provider example placed in your project root:

```yaml
schema_version: 2
name: local-coding
workspace_root: .
provider_kind: imported
model:
  source: models/local-coding-base
  device: cpu
  chat_format: auto
  allow_network: false
  max_new_tokens: 256
enabled_tools: [repo.list_files, repo.read_file, repo.search, repo.apply_patch, repo.run_tests]
capabilities: [workspace.read, workspace.write, process.test]
approval_mode: interactive
trace_dir: .latticekin/traces
budgets:
  max_steps: 12
  max_tool_calls: 12
  max_wall_seconds: 120
  max_tokens: 4096
  max_output_bytes: 65536
```

Supply a reviewed, licensed local model yourself. Its tokenizer/chat template must be compatible;
a generic adapter does not guarantee valid actions or useful repairs. Tool schemas, policy and
approval decide effects. Read [agent-platform-architecture.md](docs/agent-platform-architecture.md)
for isolation, replay and capability limits.

## Verification

Install Node 24 and run `npm ci` from `web/` before the offline repository check.
The Python suite includes genuine frontend workflow tests.

```text
python scripts/check.py
python scripts/check.py --web
python scripts/check.py --native
python scripts/check.py --package
```

`make test` delegates to the portable Python check. Checks do not install dependencies or fetch
models. Rust offline tests require dependencies already fetched; the check script handles Python
DLL discovery on Windows. CI defines Linux/Windows Python, web, locked Rust, package and container
jobs. Configuring CI is not evidence those remote jobs have already run.

## Privacy and limitations

Model outputs can be wrong or unsafe. Keep servers localhost-only unless the required authentication,
authorization and request-limit policy is explicitly configured. Trusted-tool process isolation
supports deadlines and cancellation; it is not an OS filesystem/network sandbox or tenant boundary.

Pattern-based PII filtering is limited; it does not prove a corpus contains no secrets or personal
facts. Keep private data, weights, indexes, credentials and traces out of source control.
Docker build contexts and release archives use source allowlists. Inspect artifact origin,
base/teacher and tokenizer identities; legacy artifacts can have incomplete provenance.

A passing JSON/lexical-grounding case does not prove semantic faithfulness or safe execution.
Measure held-out tasks, failure recovery, latency and memory on your actual model/hardware.

MIT license: [LICENSE.md](LICENSE.md).
