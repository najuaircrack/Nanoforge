# Agent platform architecture

The project goal is a local coding/workspace loop with explicit context, validated actions,
human approval, test feedback and replayable evidence. Model capability and executor permission
are separate decisions.

## Current public path

Project schema 2 accepts `provider_kind: manual`, `native` or `imported`;
schema 1 manifests remain readable. A local model source is relative to the manifest/project.
Imported models preserve compatible tokenizer/chat formatting; GGUF formats may require an
explicit known chat template. No generic adapter guarantees all models produce useful actions.

The manifest records model selection, enabled repository tools, capabilities, approval mode,
resource/context/token/time/output budgets, trace directory and optional evaluation suite.
The checked-in example is read-only. `approval_mode: interactive` can request a concrete write/test
approval; enabling a provider grants no additional capabilities. Sensitive/ignored/runtime files
remain excluded unless their separate read-sensitive capability is explicitly selected.

The TUI can show requests, proposed actions, outputs, remaining budgets and trace.
The studio has separate operator/artifact/evaluation surfaces. End-to-end real-model task
success must be established by executable held-out cases; availability of the views is not that evidence.

## Executor boundary

Actions are bounded versioned objects validated against a declared schema dialect.
Each tool has an explicit capability/approval policy, timeout, output budget and effect classification.
Policy-required approval cannot be erased by tool metadata.

Trusted callables are serialized with cloudpickle and run in spawned worker processes.
Cancellation/timeouts terminate the worker/process tree with platform-specific controls:
Windows Job Objects can also impose configured tree memory limits; POSIX process groups and
per-process address-space limits provide a different, more limited boundary.
Neither is an OS filesystem/network sandbox. Untrusted repository tests require a separately
configured container/OS sandbox before they can be treated as safely isolated.

Token and context budgets are distinct from output bytes. Provider parsing/input limits reject
oversized actions before execution. Memory/cost/performance outcomes depend on the actual provider
and platform; do not turn estimates or unsupported resource controls into guarantees.

## Trace, privacy and replay

Persistent checksummed traces redact payloads and argument values by default.
The trace records control transitions and reviewed metadata. Users must deliberately configure
any less-redacted mode and retain/delete traces according to their policy.
`.latticekin/` is private runtime state excluded from source/archive/container contexts.

Dry-run replay reconstructs recorded control state without rerunning tests or patches.
A trace can show that the harness followed its policy; it cannot independently prove model
semantic correctness or that an external sandbox prevented every OS side effect.

## Lifecycle and remaining validation

A capable local base plus retrieval is the practical initial quality path. The imported
SFT/LoRA API supports reviewed behavior adaptation; QLoRA requires its extra and real hardware tests.
Keep changing facts in governed retrieval/memory rather than automatic foundation retraining.

Release acceptance still needs real-model held-out repairs, context overflow, denial/cancellation,
test-failure recovery, unchanged unrelated files, trace/replay comparison and measured latency/memory.
Network credentials and tenant boundaries remain separate capabilities. Public sharing requires
explicit authentication, authorization, request limits and tested isolation.
See [roadmap.md](roadmap.md) for each unresolved feature or validation gate.
