# Coding-agent architecture

Nanoforge is now being shaped as a coding-agent platform rather than only a next-token model
trainer. The model is one component; the useful system is:

```text
repository state
  → bounded inspection
  → model proposal
  → structured tool action
  → schema/capability/approval policy
  → bounded execution
  → test observation
  → trace and replay
  → final patch or explicit failure
```

## What “frontier-like” means here

The current frontier coding systems are not made capable by a magical alternative to language
modeling. Their advantage comes from a combination of:

- large pretrained code/text/math models;
- code-specific tokenization and repository formatting;
- fill-in-the-middle (FIM) training;
- repository-level context and dependency ordering;
- long-context continuation training;
- instruction and preference tuning;
- executable tests and environment feedback;
- specialized agent-computer interfaces;
- efficient quantized inference and tool parsers.

Nanoforge cannot reproduce Qwen, DeepSeek, or StarCoder scale on an ordinary CPU. It can provide a
strong local adaptation and agent harness around an imported code model, while still supporting
small from-scratch specialists for narrow languages or DSLs.

## Model tracks

### Imported coding base — quality track

Use a compatible pretrained model such as Qwen-Coder, DeepSeek-Coder, StarCoder2, or another
licensed code model. Preserve its tokenizer, chat template, FIM token IDs, context limit,
quantization format, and action parser. Adapt with an established LoRA/QLoRA backend where the
hardware supports it. Keep repository facts in retrieval rather than memorizing volatile code.

### Small code specialist — local training track

For CPU or small GPU experiments:

- use a real BPE/SentencePiece tokenizer;
- train on one or a few languages;
- include FIM examples and ordinary causal examples;
- split by repository/snapshot, not random lines;
- filter secrets, generated files, licenses, PII, and malware-like content;
- evaluate syntax, tests, repair, and exact edits;
- describe the result as a specialist, not a general coding model.

Nanoforge now supports deterministic FIM formatting through `mode: fim` and the sentinel tokens
`<|fim_prefix|>`, `<|fim_suffix|>`, and `<|fim_middle|>`. Real training should use a tokenizer
that stores these as dedicated special tokens; the byte tokenizer path is only a smoke test.

## Agent-computer interface

The new `src/nanoforge/agent/` package provides a backend-only safe boundary:

- `ToolSpec` — description, input schema, capabilities, timeout, output limit, approval policy,
  and side-effect classification;
- `ToolRegistry` — duplicate-safe structured registration;
- `ToolExecutor` — schema validation, capability checks, approval, budgets, cancellation,
  timeout, bounded output, and trace events;
- `RepositoryWorkspace` — repository-relative file listing, reading, regex search, expected-hash
  patching, and one fixed pytest profile;
- `AgentHarness` — bounded model/action loop that accepts `tool_call` or `final` actions;
- `ReplayRunner` — dry-run trace replay that never reapplies patches or reruns tests.

The default policy allows read-only inspection. Writes and tests require explicit capabilities and
approval. The harness never accepts arbitrary model-generated shell commands.

This is a bounded process boundary, not yet a hardened OS/container sandbox. Network, credential,
arbitrary subprocess, and filesystem-outside-workspace access remain unavailable by default.

## Coding-agent evaluation

Coding evaluation must measure more than text similarity. A useful local suite should include:

1. repository inspection accuracy;
2. correct file selection;
3. safe path refusal;
4. patch correctness and stale-hash refusal;
5. approval-policy compliance;
6. fixed-profile test execution;
7. test failure observation;
8. repair after test failure;
9. no unrelated file changes;
10. trace completeness and replay equivalence.

LiveCodeBench supports the broader lesson that code models need execution, repair, and test-output
evaluation rather than only static completion. SWE-agent supports the interface lesson: an agent
needs a carefully designed computer interface for repository navigation, editing, and testing.

## Research basis

- [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186) — code/text/math mixture, FIM, repository
  continuation, long-context stages, and code-focused post-training.
- [DeepSeek-Coder](https://arxiv.org/abs/2401.14196) — repository dependency ordering, FIM,
  large-scale code data, and staged context training.
- [DeepSeek-Coder-V2](https://arxiv.org/abs/2406.11931) — continued pretraining of an MoE base,
  broad code coverage, and long-context scaling.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — expert specialization and shared experts;
  not a justification for Nanoforge’s current experimental MoE being called frontier-grade.
- [StarCoder2 / The Stack v2](https://arxiv.org/abs/2402.19173) — transparent source identity,
  repository data governance, and large-scale code training.
- [FIM](https://arxiv.org/abs/2207.14255) — infilling as a training transformation that can
  coexist with ordinary causal generation.
- [LiveCodeBench](https://arxiv.org/abs/2403.07974) — contamination-resistant code evaluation,
  execution, repair, and test-output prediction.
- [SWE-agent](https://arxiv.org/abs/2405.15793) — agent-computer interfaces for repository
  navigation, editing, and testing.

## Explicit non-goals for this slice

- no claim of reproducing a frontier model’s weights or benchmark scores;
- no arbitrary shell execution;
- no automatic network access;
- no autonomous source modification without approval;
- no semantic code correctness claim from JSON validity alone;
- no QLoRA claim from the existing CPU/int8 path;
- no hardened container sandbox claim yet.
