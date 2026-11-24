from __future__ import annotations

import json

import pytest

from nanoforge.cli import main


def test_validate_config_is_strict_by_default(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("training:\n  misspelled: 7\n")
    with pytest.raises(ValueError, match="unknown"):
        main(["validate-config", "--config", str(config)])


def test_serving_cli_forwards_resource_and_auth_policy(monkeypatch):
    import nanoforge.server
    captured = {}
    monkeypatch.setenv("LATTICEKIN_TEST_API_KEY", "synthetic-key-123456789")
    def serve(*args, **kwargs):
        captured.update(kwargs)
    monkeypatch.setattr(nanoforge.server, "serve", serve)
    main(["serve", "--checkpoint", "synthetic.pt", "--api-key-env", "LATTICEKIN_TEST_API_KEY",
          "--workspace", ".", "--max-concurrency", "3", "--request-timeout", "5"])
    assert captured["api_key"] == "synthetic-key-123456789"
    assert captured["max_concurrency"] == 3
    assert captured["request_timeout"] == 5


def test_adaptation_cli_invokes_real_config_contract(tmp_path, monkeypatch):
    import nanoforge.training.pretrained
    config = tmp_path / "adapt.json"
    config.write_text(json.dumps({"source": "local-base", "data_path": "public.jsonl",
                                 "output_dir": "output", "max_steps": 1}))
    captured = []
    monkeypatch.setattr(nanoforge.training.pretrained, "train_pretrained",
                        lambda config: captured.append(config) or {"status": "completed"})
    main(["adapt", "--config", str(config)])
    assert captured[0].allow_network is False
    assert captured[0].max_steps == 1


def test_native_sampling_controls_reach_generation(monkeypatch, capsys):
    import nanoforge.cli
    calls = []
    class Engine:
        def complete(self, prompt, tokens, sampling, stop_tokens, **kwargs):
            calls.append(sampling)
            return "result"
    monkeypatch.setattr(nanoforge.cli, "_load_generation_engine", lambda args: Engine())
    main(["generate", "--checkpoint", "synthetic", "--sampler", "contrastive",
          "--contrastive-top-k", "3"])
    assert calls[0].sampler == "contrastive"
    assert calls[0].contrastive_top_k == 3
    assert "result" in capsys.readouterr().out


def test_chat_keeps_causal_multiturn_history(monkeypatch):
    import nanoforge.cli
    prompts = iter(["first", "second", ""])
    calls = []
    class Engine:
        def chat_stream(self, messages, *args, **kwargs):
            calls.append(list(messages))
            yield "answer"
    monkeypatch.setattr(nanoforge.cli, "_load_generation_engine", lambda args: Engine())
    monkeypatch.setattr("builtins.input", lambda text: next(prompts))
    main(["chat", "--checkpoint", "synthetic"])
    assert calls[0] == [("user", "first")]
    assert calls[1] == [("user", "first"), ("assistant", "answer"), ("user", "second")]


def test_project_run_and_saved_replay_are_public_commands(tmp_path, capsys):
    project = tmp_path / "project.yaml"
    project.write_text("schema_version: 2\nname: synthetic-cli\nworkspace_root: .\nprovider_kind: manual\n")
    actions = tmp_path / "actions.json"
    actions.write_text(json.dumps([{"schema_version": 1, "type": "final", "content": "synthetic result"}]))
    main(["agent-run", "--project", str(project), "--request", "synthetic task",
          "--actions", str(actions)])
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed"
    assert output["answer"] == "synthetic result"
    trace = tmp_path / output["trace_path"]
    assert trace.exists()
    main(["agent-replay", "--trace", str(trace)])
    replay = json.loads(capsys.readouterr().out)
    assert replay["side_effects_applied"] is False
    assert replay["trace_complete"] is True


def test_project_run_reports_tool_policy_denial(tmp_path, capsys):
    project = tmp_path / "project.yaml"
    project.write_text("schema_version: 2\nname: denial\nworkspace_root: .\nprovider_kind: manual\n"
                       "enabled_tools: [repo.apply_patch]\ncapabilities: [workspace.read]\n")
    actions = tmp_path / "actions.json"
    actions.write_text(json.dumps([
        {"schema_version": 1, "type": "tool_call", "name": "repo.apply_patch",
         "arguments": {"path": "synthetic.py", "expected_sha256": None, "replacement_text": "value = 1\n"}},
        {"schema_version": 1, "type": "final", "content": "denied"}]))
    main(["agent-run", "--project", str(project), "--request", "synthetic task",
          "--actions", str(actions)])
    result = json.loads(capsys.readouterr().out)
    assert result["observations"][0]["status"] == "denied"
    assert list(tmp_path.glob("*.py")) == []


def test_native_json_grammar_reaches_generation(monkeypatch):
    import nanoforge.cli
    from nanoforge.data.tokenizer import ByteTokenizer
    calls = []
    class Engine:
        tokenizer = ByteTokenizer()
        def complete(self, *args, **kwargs):
            calls.append(kwargs)
            return "{}"
    monkeypatch.setattr(nanoforge.cli, "_load_generation_engine", lambda args: Engine())
    main(["generate", "--checkpoint", "synthetic", "--grammar", "json"])
    assert calls[0]["constraint"].is_accepting(Engine.tokenizer.encode("{}"))


def test_compare_reports_rejects_candidate_regressions(tmp_path, capsys):
    report = {"schema_version": 1, "report_type": "task_suite", "report_id": "baseline",
              "suite_id": "synthetic", "suite_fingerprint": "same-suite",
              "cases": [{"case_id": "case", "status": "passed", "latency_ms": 1}],
              "summary": {"pass_rate": 1.0}}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(report))
    report.update(report_id="candidate", cases=[{"case_id": "case", "status": "failed", "latency_ms": 1}],
                  summary={"pass_rate": 0.0})
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(report))
    with pytest.raises(SystemExit) as error:
        main(["compare-evaluations", str(baseline), str(candidate)])
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["candidates"][0]["regressions"] == ["case"]

