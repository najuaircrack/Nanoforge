from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nanoforge.artifacts import ArtifactRegistry, ResourceMonitor, migrate_manifest
from nanoforge.config import ModelConfig, MoEConfig, NanoforgeConfig
from nanoforge.evaluation.catalog import compare_evaluation_reports, load_evaluation_report
from nanoforge.evaluation.suite import (
    SnapshotPolicy,
    TaskCase,
    TaskSuite,
    check_grounding,
    check_structured_output,
    check_tool_call,
    evaluate_task_suite,
    load_task_suite,
    write_evaluation_report,
)
from nanoforge.rag.memory import MemoryStore


class Generator:
    def __init__(self, output="ok"):
        self.output = output
        self.prompts = []

    def complete(self, prompt, max_new_tokens, sampling, stop_tokens=None):
        self.prompts.append(prompt)
        return self.output


def _report(case_id="one", output="ok", expected="ok"):
    return evaluate_task_suite(
        TaskSuite("fixture", [TaskCase(case_id, "exact_match", "prompt", expected)]),
        Generator(output), device="cpu", seed=7,
    )


def test_context_and_effective_options_are_snapshotted():
    generator = Generator("fact [d1]")
    case = TaskCase("g", "grounding", "Use evidence", {
        "required_citations": ["d1"], "required_substrings": ["fact"],
    }, context=[{"id": "d1", "text": "fact"}], generation={"max_new_tokens": 9})
    report = evaluate_task_suite(TaskSuite("g", [case]), generator, device="cpu")
    assert "fact" in generator.prompts[0] and "d1" in generator.prompts[0]
    assert report.cases[0].snapshot["effective_prompt"] == generator.prompts[0]
    assert report.cases[0].snapshot["generation"]["max_new_tokens"] == 9


def test_snapshot_redaction_applies_to_inputs_outputs_and_checks():
    report = evaluate_task_suite(TaskSuite("p", [TaskCase("p", "exact_match", "private-fixture", "private-fixture")]),
                                 Generator("private-fixture"), device="cpu",
                                 snapshot_policy=SnapshotPolicy(sensitive_values=("private-fixture",)))
    assert "private-fixture" not in json.dumps(report.to_dict())
    assert report.summary["passed"] == 1
    hidden = evaluate_task_suite(TaskSuite("p", [TaskCase("p", "exact_match", "secret", "ok")]),
                                Generator(), device="cpu", snapshot_policy=SnapshotPolicy(include_inputs=False))
    assert "prompt" not in hidden.cases[0].snapshot
    assert hidden.cases[0].snapshot["input_fingerprint"]


@pytest.mark.parametrize("raw", [
    {"id":"x","type":"unknown","prompt":"x","expected":"x"},
    {"id":"x","type":"exact_match","prompt":"x","expected":"x","generation":{"temprature":1}},
    {"id":"x","type":"exact_match","prompt":"x","expected":"x","generation":{"max_new_tokens":"bad"}},
    {"id":"x","type":"grounding","prompt":"x","expected":{}},
    {"id":"x","type":"tool_call","prompt":"x","expected":{"name":"x"}},
])
def test_suite_loader_rejects_invalid_cases_with_line_number(tmp_path, raw):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_task_suite(path)


def test_direct_invalid_case_remains_visible_as_error():
    report = evaluate_task_suite(TaskSuite("x", [TaskCase("bad", "exact_match", "x", "x",
                                                           generation={"max_new_tokens":"bad"})]), Generator(), device="cpu")
    assert report.summary["errors"] == 1


def test_strict_json_checkers_do_not_pass_bool_missing_keys_or_vacuous_grounding():
    assert not check_structured_output('{"count":true}', {"count":1}).passed
    assert not check_structured_output('{"a":1,"a":2}', {"a":2}).passed
    assert not check_tool_call('{"name":"read"}', {"name":"read","arguments":{}}).passed
    assert not check_grounding("unsupported", {}).passed
    assert not check_grounding("doc-10", {"required_citations":["doc-1"]}).passed


def test_reports_are_independent_and_authoritative(tmp_path):
    first = write_evaluation_report(_report("A"), tmp_path / "a.json")
    second = write_evaluation_report(_report("B"), tmp_path / "b.json")
    assert first["cases"] != second["cases"]
    assert json.loads(first["cases"].read_text())["case_id"] == "A"
    assert load_evaluation_report(first["report"])["cases"][0]["case_id"] == "A"
    with pytest.raises(FileExistsError):
        write_evaluation_report(_report(), tmp_path / "a.json")
    first["cases"].write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_evaluation_report(first["report"])


def test_comparison_gates_regressions_and_refuses_different_suites(tmp_path):
    baseline = write_evaluation_report(_report(), tmp_path / "a.json")["report"]
    candidate = write_evaluation_report(_report(output="wrong"), tmp_path / "b.json")["report"]
    comparison = compare_evaluation_reports([baseline, candidate])
    assert comparison["compatible"]
    assert comparison["candidates"][0]["regressions"] == ["one"]
    assert not comparison["candidates"][0]["thresholds_passed"]
    other = write_evaluation_report(_report(expected="other"), tmp_path / "c.json")["report"]
    with pytest.raises(ValueError, match="suite"):
        compare_evaluation_reports([baseline, other])


def test_masked_metrics_use_only_supervised_tokens(monkeypatch):
    from types import SimpleNamespace
    import nanoforge.evaluation.metrics as metrics

    class Model:
        def __init__(self, config): pass
        def load_state_dict(self, *args, **kwargs): pass
        def to(self, *args): return self
        def eval(self): return self
        def __call__(self, x, labels):
            logits = torch.zeros((1,4,3)); logits[:,:,1] = 1
            return SimpleNamespace(loss=torch.tensor(0.1), logits=logits)

    monkeypatch.setattr(metrics, "load_checkpoint", lambda *a,**k:{"config":NanoforgeConfig(),"model":{}})
    monkeypatch.setattr(metrics, "NanoforgeForCausalLM", Model)
    monkeypatch.setattr(metrics, "PackedMemmapDataset", lambda *a:object())
    monkeypatch.setattr(metrics, "make_torch_batch", lambda *a:(torch.zeros(1,4,dtype=torch.long),torch.tensor([[1,-100,-100,-100]])))
    result = metrics.evaluate_checkpoint("fixture", "fixture", 4, batches=1, device="cpu")
    assert result["token_accuracy"] == 1.0 and result["evaluated_tokens"] == 1
    with pytest.raises(ValueError): metrics.evaluate_checkpoint("fixture","fixture",4,batches=0)


def test_profile_counts_actual_moe_parameters():
    from nanoforge.model.transformer import NanoforgeModel
    from nanoforge.profiling import estimate_model_profile
    config = ModelConfig(vocab_size=128, max_seq_len=16, d_model=32, n_layers=2, n_heads=4,
                         moe=MoEConfig(num_experts=4,top_k=2))
    assert estimate_model_profile(config).parameters == sum(p.numel() for p in NanoforgeModel(config).parameters())


def test_index_is_mmap_transactional_and_validated(tmp_path):
    from nanoforge.rag import load_index
    from nanoforge.rag.embeddings import TfIdfEmbedder
    from nanoforge.rag.store import NumpyVectorStore
    embedder = TfIdfEmbedder(vocab_size=16); embedder.fit(["alpha", "beta"])
    store = NumpyVectorStore.build_and_save(tmp_path, embedder, iter([("alpha","fixture",{}),("beta","fixture",{})]),
                                            "tfidf", embedder_state=embedder.to_dict(),batch_size=1)
    assert isinstance(store.vectors, np.memmap)
    assert not isinstance(store.docs, list)
    assert load_index(tmp_path).query("alpha",top_k=1)[0]["text_chunk"] == "alpha"
    commit = json.loads((tmp_path / "index_meta.json").read_text())
    (tmp_path / commit["files"]["docs"]).write_text("bad",encoding="utf-8")
    with pytest.raises(ValueError,match="integrity"):
        NumpyVectorStore.load(tmp_path)


def test_sentence_transformer_identity_roundtrips_and_missing_model_fails(tmp_path,monkeypatch):
    import nanoforge.rag.embeddings as embeddings
    from nanoforge.rag import build_index,load_index
    names=[]
    class Encoder:
        def __init__(self,name="all-MiniLM-L6-v2"):
            self.model_name=name;self.dim=2;names.append(name)
        def embed_batch(self,texts):return np.tile([[1.,0.]],(len(texts),1)).astype(np.float32)
    monkeypatch.setattr(embeddings,"StEmbedder",Encoder)
    source=tmp_path/"fixture.txt";source.write_text("alpha beta",encoding="utf-8")
    build_index([source],tmp_path/"idx",backend="sentence-transformers",st_model="custom")
    assert load_index(tmp_path/"idx").query("alpha")[0]["score"] == 1.0
    assert names==["custom","custom"]
    def unavailable(*a,**k):raise ImportError("unavailable")
    monkeypatch.setattr(embeddings,"StEmbedder",unavailable)
    with pytest.raises(ImportError):load_index(tmp_path/"idx")


def test_memory_requires_consent_owner_scope_and_retention(tmp_path):
    memory=MemoryStore(tmp_path/"memory.sqlite",owner="alice")
    with pytest.raises(PermissionError):memory.put("private fixture",scope="project-a",provenance={"source":"test"},personal=True)
    record=memory.put("private fixture",scope="project-a",provenance={"source":"test"},personal=True,consent=True,retention_seconds=60)
    assert memory.get(record.memory_id,scope="project-a").content=="private fixture"
    assert memory.get(record.memory_id,scope="project-b") is None
    assert MemoryStore(tmp_path/"memory.sqlite",owner="bob").get(record.memory_id,scope="project-a") is None
    assert memory.export_training(scope="project-a",consent=True)==[]
    assert memory.delete(record.memory_id,scope="project-a")
    assert memory.get(record.memory_id,scope="project-a") is None
    assert all("private fixture" not in json.dumps(event) for event in memory.audit_events())


def test_memory_expiry_and_explicit_training_selection(tmp_path):
    now=[100.0];memory=MemoryStore(tmp_path/"m.sqlite",owner="alice",clock=lambda:now[0])
    record=memory.put("synthetic text",scope="a",provenance={"source":"fixture"},consent=True,training_allowed=True,retention_seconds=1)
    assert len(memory.export_training(scope="a",consent=True))==1
    now[0]=102
    assert memory.get(record.memory_id,scope="a") is None
    assert memory.purge_expired()==1


def test_schema_zero_migrates_and_invalid_versions_fail():
    assert migrate_manifest({"schema_version":0})["schema_version"]==1
    for version in [-1,True,"1"]:
        with pytest.raises(ValueError):migrate_manifest({"schema_version":version})


def test_artifact_registry_is_content_bound_and_immutable(tmp_path):
    source=tmp_path/"input.txt";source.write_text("synthetic",encoding="utf-8")
    registry=ArtifactRegistry(tmp_path/"registry")
    registered=registry.register(source,artifact_kind="fixture",owner="alice")
    assert registry.verify(registered.artifact_id)
    payload=registry.path_for(registered.artifact_id)
    payload.write_text("tampered",encoding="utf-8")
    with pytest.raises(ValueError,match="integrity"):registry.verify(registered.artifact_id)


def test_resource_monitor_labels_sampling_and_process_scope():
    with ResourceMonitor(interval=0.01) as monitor:
        allocation=bytearray(128*1024)
        assert allocation
    assert monitor.result["elapsed_seconds"]>=0
    assert monitor.result["peak_is_sampled"] is True
    assert monitor.result["scope"]=="process-during-run"
