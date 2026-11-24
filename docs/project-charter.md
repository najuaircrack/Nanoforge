# Latticekin project charter

**Working public name:** Latticekin
**Repository compatibility name:** Nanoforge
**Status:** product definition, October 2026

Latticekin is a local-first platform for turning a model, private data, retrieval, tools,
and evaluations into useful AI agents that run on a CPU, one GPU, or a small local cluster.
It is not a promise that a small randomly initialized model can become a frontier model by
training on a few files.

## The problem

Most local AI projects stop at one of two incomplete endpoints:

1. a model can generate text, but it cannot safely use tools, inspect a workspace, recover
   from failures, or prove that it solved a task; or
2. an agent wrapper can call tools, but the model, data, prompts, traces, evaluations, and
   deployment settings are not reproducible or locally governed.

People need one inspectable loop that connects:

```text
model → private context → adaptation → structured action → policy → tool → observation
       → evaluation → replayable deployment
```

The hard product problem is therefore not “add another transformer.” It is making the whole
agent loop measurable, bounded, private by default, and useful on ordinary hardware.

## Who it is for

- developers who want a private coding or workspace agent;
- researchers studying small models, adaptation, retrieval, tool use, and evaluation;
- teams with a CPU-only or modest GPU workstation and local data;
- educators who need a readable, reproducible training and agent harness;
- advanced users who want to import a capable model and keep the agent state local.

## The first public use case

The first release targets a **local coding/workspace agent**. A user provides a repository
and a request. The intended complete workflow inspects files, searches the workspace and proposes a repository-relative
patch, asks for approval, runs a fixed test profile, and explains the result. Every action is
bounded and recorded in a redacted trace.

This is a deliberately narrow wedge because coding has executable feedback: a patch can be
tested, replayed, compared, and rejected. The same harness can later support research, data,
operations, and personal knowledge workflows without pretending that all domains are solved.

## What Latticekin promises

- **Local first:** model files, private data, indexes, and traces stay on the user machine by
  default.
- **Model agnostic:** imported models and Latticekin-trained models use the same agent protocol.
- **Adaptation first:** retrieval and small, curated adapters teach local behavior; raw private
  data is not silently treated as general pretraining data.
- **Evaluation first:** success means held-out task quality, safe tool use, latency, memory, and
  failure recovery—not only lower training loss.
- **Human controlled:** the model proposes actions; policy, approval, budgets, and the executor
  decide what can happen.
- **Inspectable:** requests, context, actions, tool results, approvals, failures, and final
  answers are visible and replayable.
- **CPU and GPU practical:** CPU is a supported quality/performance target, not an afterthought;
  GPU users get larger models and faster adapter experiments.

## What it does not promise

Latticekin will not claim any of the following without measured evidence:

- frontier intelligence from a 10–50M parameter scratch model;
- broad knowledge learned from a small personal corpus;
- useful capability merely because a 50-second training run completed;
- twenty independent trainer processes as a substitute for synchronized distributed training;
- a production quantized training or inference kernel when it is only a fallback;
- arbitrary model-generated shell, Python, network, credential, or filesystem access;
- autonomous self-modification without human review;
- public multi-user serving before authentication, authorization, isolation, and request limits.

Scratch training remains valuable for education, controlled experiments, smoke tests, narrow
DSLs, and small specialists. It is not the default quality path for a general assistant.

## The practical quality path

For a user who wants a capable local agent, the default path is:

```text
licensed pretrained model
  + quantized local inference
  + retrieval for changing/private knowledge
  + curated SFT or LoRA behavior adaptation
  + structured tools and approvals
  + executable task evaluation
  + optional distillation into a smaller specialist
```

The platform may train its own compact models, but those models must carry honest provenance:
scratch-trained, imported, adapted, distilled, or composed. A local model can be excellent at a
narrow task without being a general frontier model.

## Product vocabulary

- **Model:** a language model or adapter used for inference or training.
- **Project:** a workspace, model selection, data sources, policy, tools, evaluations, and
  resource budgets stored in a versioned manifest.
- **Harness:** the state machine that turns a request into bounded model actions and observations.
- **Tool:** a schema-validated capability with an explicit timeout, output limit, approval policy,
  and audit event.
- **Run:** a reproducible execution with a manifest, metrics, trace, failures, and artifacts.
- **Memory:** governed state with scope, owner, provenance, retention, and deletion behavior.

## First release acceptance criteria

The first public release is useful when a user can:

1. install the project and choose an imported local model or a small built-in smoke model;
2. create a project manifest for a repository and a task suite;
3. launch the custom terminal harness;
4. inspect the selected context and structured model action;
5. allow read-only tools by default and approve writes/tests explicitly;
6. see bounded tool output, failures, remaining budgets, and a live trace;
7. replay the trace without reapplying side effects;
8. compare the run against held-out task, latency, memory, and safety results.

The current repository has the beginnings of these pieces: data preparation, compact training,
model import paths, RAG primitives, task evaluation, a Textual training UI, and a bounded backend
harness. The next work is integration, not a claim that the whole platform already exists.

## Naming decision

The public name is **Latticekin**: a lattice of model, context, tools, policy, and evaluation,
with “kin” emphasizing a local, human-governed companion rather than an autonomous authority.

On October 15, 2026, quick availability checks found no exact GitHub repository or user, no exact
DuckDuckGo result, no PyPI project, and no npm package for `latticekin`. This is not trademark,
company-name, domain, or legal clearance. Before publishing, check those separately and reserve
the desired handles. Until the namespace migration is complete, `nanoforge` remains the Python
module and compatibility CLI name.
