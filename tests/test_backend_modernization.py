from __future__ import annotations

import json
import time

import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from nanoforge.server import create_app as inference_app
from nanoforge.webapp import create_app as studio_app


class TestEngine:
    __test__ = False

    def stream(self, prompt, max_new_tokens=256, sampling=None, **kwargs):
        yield 'hello'
        yield ' world'

    def complete(self, prompt, max_new_tokens=256, sampling=None, **kwargs):
        return ''.join(self.stream(prompt, max_new_tokens, sampling, **kwargs))

    def count_tokens(self, text):
        return len(text.split())

    def chat_stream(self, messages, **kwargs):
        yield from self.stream('synthetic')


def test_inference_json_and_openai_envelopes():
    with TestClient(inference_app(TestEngine())) as client:
        response = client.post('/v1/completions', json={'prompt': 'question', 'max_tokens': 8})
        assert response.status_code == 200
        result = response.json()
        assert result['choices'][0]['text'] == 'hello world'
        assert result['usage']['completion_tokens'] == 2
        response = client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'question'}]})
        assert response.status_code == 200
        assert response.json()['choices'][0]['message']['content'] == 'hello world'
        assert client.get('/metrics').json()['nanoforge_generated_tokens_total'] == 4


def test_openai_stream_matches_nonstream_usage():
    with TestClient(inference_app(TestEngine())) as client:
        response = client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'q'}], 'stream': True})
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/event-stream')
        assert 'chat.completion.chunk' in response.text and 'data: [DONE]' in response.text
        assert client.get('/metrics').json()['nanoforge_generated_tokens_total'] == 2


def test_upload_is_transactional_confined_and_collision_safe(tmp_path):
    with TestClient(studio_app('runs/test', workspace_root=tmp_path)) as client:
        response = client.post('/api/data/upload', files={'file': ('sample.txt', b'first')})
        assert response.status_code == 200 and response.json()['ok']
        assert (tmp_path / response.json()['path']).read_bytes() == b'first'
        collision = client.post('/api/data/upload', files={'file': ('sample.txt', b'second')})
        assert collision.status_code == 409
        assert (tmp_path / 'data/uploads/sample.txt').read_bytes() == b'first'
        for escape in ('../outside', '/outside', r'C:\outside', r'\\server\share'):
            response = client.post('/api/data/upload', data={'subdir': escape}, files={'file': ('sample2.txt', b'x')})
            assert response.status_code in (400, 403)


def test_config_failed_save_keeps_original_and_paths_are_confined(tmp_path):
    target = tmp_path / 'config.yaml'
    target.write_text('model: {}\n', encoding='utf-8')
    with TestClient(studio_app('runs/test', workspace_root=tmp_path)) as client:
        response = client.post('/api/config/save', json={'path': 'config.yaml', 'yaml_text': 'model: ['})
        assert response.json()['ok'] is False
        assert target.read_text() == 'model: {}\n'
        response = client.get('/api/config/raw', params={'path': str(tmp_path.parent / 'outside.txt')})
        assert response.status_code in (400, 403)


def test_bad_chat_and_generation_numbers_are_validation_errors(tmp_path):
    with TestClient(studio_app('runs/test', workspace_root=tmp_path)) as client:
        assert client.post('/api/chat/stream', json={'messages': [['user']]}).status_code == 422
        assert client.post('/api/chat/generate', json={'prompt': 'q', 'max_new_tokens': -1}).status_code == 422
        assert client.post('/api/chat/generate', json={'prompt': 'q', 'temperature': -1}).status_code == 422


def test_download_uses_arguments_and_auto_train_matches_cli(tmp_path, monkeypatch):
    from nanoforge.serving import JobManager
    commands = []
    monkeypatch.setattr(JobManager, 'start_process', lambda self, command, **kw: commands.append((command, kw)) or 'job')
    with TestClient(studio_app('runs/test', workspace_root=tmp_path)) as client:
        response = client.post('/api/jobs/start', json={'type': 'download', 'dataset': "test');print('injected')"})
        assert response.json()['ok']
        command = commands[-1][0]
        assert '-c' not in command and '--source' in command
        assert command[command.index('--source') + 1] == "test');print('injected')"
        response = client.post('/api/jobs/start', json={'type': 'auto-train', 'input': ['data/sample.txt'], 'name': 'specialist'})
        assert response.json()['ok']
        command = commands[-1][0]
        assert '--config' not in command and '--input' in command and '--name' in command


def test_authentication_origin_and_request_size(tmp_path):
    from nanoforge.serving import ServingPolicy
    with pytest.raises(ValueError):
        ServingPolicy(host='0.0.0.0')
    policy = ServingPolicy(host='0.0.0.0', api_key='synthetic-key-123456', max_request_bytes=256)
    with TestClient(studio_app('runs/test', workspace_root=tmp_path, policy=policy)) as client:
        assert client.get('/api/jobs').status_code == 401
        headers = {'Authorization': 'Bearer synthetic-key-123456'}
        assert client.get('/api/jobs', headers=headers).status_code == 200
        assert client.post('/api/jobs/start', headers={**headers, 'Origin': 'https://untrusted.example'}, json={'type': 'download', 'dataset': 'sample'}).status_code == 403
        assert client.post('/api/chat/generate', headers=headers, json={'prompt': 'x' * 300}).status_code == 413


def test_engine_deadline_retains_concurrency_until_worker_settles():
    from nanoforge.serving import EngineRunner, ServingPolicy
    class Slow(TestEngine):
        def stream(self, *args, **kwargs):
            time.sleep(.15)
            yield 'done'
    runner = EngineRunner(ServingPolicy(request_timeout=.02, max_concurrency=1))
    with pytest.raises(TimeoutError):
        list(runner.stream(Slow(), 'q', 2))
    with pytest.raises(RuntimeError, match='busy'):
        list(runner.stream(Slow(), 'q', 2))
    time.sleep(.2)
    assert runner.active == 0


def test_job_logs_lifecycle_and_retention(tmp_path):
    import sys

    from nanoforge.serving import JobManager
    manager = JobManager(tmp_path, max_log_lines=3, max_jobs=2, deadline_seconds=2)
    job_id = manager.start_process([sys.executable, '-c', "for i in range(8): print(i)"], job_type='test')
    deadline = time.monotonic() + 5
    while not manager.status(job_id)['done'] and time.monotonic() < deadline:
        time.sleep(.02)
    assert manager.status(job_id)['done']
    assert manager.status(job_id)['code'] == 0
    assert manager.status(job_id)['lines'] == ['5', '6', '7']
    assert manager.status(job_id)['logs_truncated']
    manager.close()


def test_studio_instances_do_not_share_models_and_import_loader(tmp_path, monkeypatch):
    import nanoforge.imports as imports
    (tmp_path / 'synthetic-import').mkdir()
    monkeypatch.setattr(imports, 'load_imported_engine', lambda source, **kwargs: TestEngine())
    with TestClient(studio_app('runs/test', workspace_root=tmp_path)) as one, TestClient(studio_app('runs/test', workspace_root=tmp_path)) as two:
        result = one.post('/api/chat/load', json={'path': 'synthetic-import'}).json()
        assert result['ok']
        assert one.get('/api/chat/status').json()['loaded'] == 'synthetic-import'
        assert two.get('/api/chat/status').json()['loaded'] is None


def test_workspace_selection_and_job_bound_metrics(tmp_path):
    root = tmp_path / 'runs/actual'
    root.mkdir(parents=True)
    (root / 'metrics.jsonl').write_text(json.dumps({'step': 7, 'train/loss': 1.25}) + '\n')
    with TestClient(studio_app('runs/other', workspace_root=tmp_path)) as client:
        result = client.get('/api/metrics', params={'run_dir': 'runs/actual'}).json()
        assert result['rows'][0]['step'] == 7
        assert client.get('/api/jobs').json()['jobs'] == []


def wait_agent(client, run_id, *, approval=False):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        run = client.get('/api/agent/runs/' + run_id).json()['run']
        if run['done'] or (approval and run['pending_approvals']):
            return run
        time.sleep(.02)
    pytest.fail('agent HTTP run did not reach expected state')


def test_project_manual_run_persistent_trace_and_replay(tmp_path):
    with TestClient(studio_app('runs/test', workspace_root=tmp_path)) as client:
        defaults = client.get('/api/projects/schema').json()['defaults']
        defaults['resources']['context_paths'] = ['sample.py']
        (tmp_path / 'sample.py').write_bytes(b'print(42)\n')
        saved = client.post('/api/projects/save', json={'path': 'project.yaml', 'project': defaults}).json()
        assert saved['ok'], saved
        assert client.get('/api/projects/context', params={'path': 'project.yaml'}).json()['files'][0]['text'] == 'print(42)\n'
        result = client.post('/api/agent/runs', json={'project_path': 'project.yaml', 'request': 'answer', 'actions': [{'schema_version': 1, 'type': 'final', 'content': 'synthetic final'}]}).json()
        assert result['ok'], result
        run = wait_agent(client, result['run_id'])
        assert run['state'] == 'completed', run
        assert run['result']['answer'] == 'synthetic final'
        trace_path = tmp_path / run['trace_path']
        assert trace_path.is_file() and 'synthetic final' not in trace_path.read_text()
        replay = client.post('/api/agent/replay', json={'path': run['trace_path']}).json()['replay']
        assert replay['trace_complete'] and replay['side_effects_applied'] is False


def test_agent_patch_requires_explicit_http_approval_and_cancel(tmp_path):
    import hashlib
    target = tmp_path / 'sample.py'
    target.write_text('old')
    with TestClient(studio_app('runs/test', workspace_root=tmp_path)) as client:
        defaults = client.get('/api/projects/schema').json()['defaults']
        defaults.update(approval_mode='interactive', enabled_tools=['repo.apply_patch'], capabilities=['workspace.read', 'workspace.write'])
        assert client.post('/api/projects/save', json={'path': 'project.yaml', 'project': defaults}).json()['ok']
        actions = [{'schema_version': 1, 'type': 'tool_call', 'name': 'repo.apply_patch', 'arguments': {'path': 'sample.py', 'expected_sha256': hashlib.sha256(b'old').hexdigest(), 'replacement_text': 'new'}}, {'schema_version': 1, 'type': 'final', 'content': 'done'}]
        started = client.post('/api/agent/runs', json={'project_path': 'project.yaml', 'request': 'patch', 'actions': actions}).json()
        run = wait_agent(client, started['run_id'], approval=True)
        assert target.read_text() == 'old' and not run['done'], run
        approval_id = run['pending_approvals'][0]['approval_id']
        assert client.post('/api/agent/runs/' + started['run_id'] + '/approve', json={'approval_id': approval_id, 'approved': True}).json()['ok']
        run = wait_agent(client, started['run_id'])
        assert run['state'] == 'completed' and target.read_text() == 'new', run['result']
        replay = client.post('/api/agent/replay', json={'run_id': started['run_id']}).json()['replay']
        assert replay['side_effects_applied'] is False and target.read_text() == 'new'
        target.write_text('old')
        started = client.post('/api/agent/runs', json={'project_path': 'project.yaml', 'request': 'patch', 'actions': actions}).json()
        run = wait_agent(client, started['run_id'], approval=True)
        client.post('/api/agent/runs/' + started['run_id'] + '/cancel')
        run = wait_agent(client, started['run_id'])
        assert run['done'] and target.read_text() == 'old'


def test_http_deadline_returns_504_then_busy_until_settled():
    from nanoforge.serving import ServingPolicy
    class Slow(TestEngine):
        def stream(self, *args, **kwargs):
            time.sleep(.2)
            yield 'done'
    with TestClient(inference_app(Slow(), policy=ServingPolicy(request_timeout=.02, max_concurrency=1))) as client:
        assert client.post('/v1/completions', json={'prompt': 'q'}).status_code == 504
        assert client.post('/v1/completions', json={'prompt': 'q'}).status_code == 429


def test_native_cpu_model_is_served_through_real_engine():
    from nanoforge.config import DataConfig, ModelConfig, NanoforgeConfig
    from nanoforge.data.tokenizer import ByteTokenizer
    from nanoforge.generation.engine import GenerationEngine
    from nanoforge.model.transformer import build_model
    tokenizer = ByteTokenizer()
    config = NanoforgeConfig(model=ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=64), data=DataConfig(tokenizer_type='byte'))
    engine = GenerationEngine(build_model(config), tokenizer, device='cpu', config=config)
    with TestClient(inference_app(engine)) as client:
        response = client.post('/v1/completions', json={'prompt': 'hi', 'max_tokens': 2, 'mode': 'deterministic'})
        assert response.status_code == 200, response.text
        assert isinstance(response.json()['choices'][0]['text'], str)
        assert isinstance(response.json()['usage']['completion_tokens'], int)


def test_evaluation_catalog_comparison_and_artifact_http(tmp_path):
    from nanoforge.artifacts import write_artifact_manifest
    from nanoforge.evaluation.suite import (
        evaluate_task_suite,
        load_task_suite,
        write_evaluation_report,
    )
    suite_path = tmp_path / 'suite.jsonl'
    suite_path.write_text(json.dumps({'id': 'case', 'type': 'exact_match', 'prompt': 'question', 'expected': 'hello world', 'generation': {'max_new_tokens': 4}}) + '\n')
    suite = load_task_suite(suite_path)
    first = write_evaluation_report(evaluate_task_suite(suite, TestEngine(), device='cpu'), tmp_path / 'reports/first')['report']
    second = write_evaluation_report(evaluate_task_suite(suite, TestEngine(), device='cpu'), tmp_path / 'reports/second')['report']
    artifact = tmp_path / 'artifact.txt'
    artifact.write_text('synthetic')
    write_artifact_manifest(artifact, artifact_kind='test')
    with TestClient(studio_app('runs/test', workspace_root=tmp_path)) as client:
        assert client.get('/api/evaluations/suites').json()['suites'][0]['count'] == 1
        reports = client.get('/api/evaluations/reports').json()['reports']
        assert len(reports) == 2
        report = client.get('/api/evaluations/report', params={'path': str(first.relative_to(tmp_path))}).json()
        assert report['report']['summary']['passed'] == 1
        comparison = client.post('/api/evaluations/compare', json={'report_paths': [str(first.relative_to(tmp_path)), str(second.relative_to(tmp_path))]}).json()['comparison']
        assert comparison['compatible'] and comparison['candidates'][0]['regressions'] == []
        artifacts = client.get('/api/artifacts').json()['artifacts']
        assert any(item['manifest']['artifact_kind'] == 'test' for item in artifacts)


def test_evaluation_job_runs_real_cpu_checkpoint_end_to_end(tmp_path):
    from nanoforge.config import DataConfig, ModelConfig, NanoforgeConfig
    from nanoforge.model.transformer import build_model
    from nanoforge.training.checkpoint import save_checkpoint
    cfg = NanoforgeConfig(model=ModelConfig(vocab_size=259, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=64), data=DataConfig(tokenizer_type='byte'))
    save_checkpoint(tmp_path / 'tiny.pt', build_model(cfg), None, cfg, 0)
    suite = tmp_path / 'suite.jsonl'
    suite.write_text(json.dumps({'id': 'case', 'type': 'exact_match', 'prompt': 'hi', 'expected': 'known', 'generation': {'max_new_tokens': 2, 'mode': 'deterministic'}}) + '\n')
    with TestClient(studio_app('runs/test', workspace_root=tmp_path)) as client:
        started = client.post('/api/evaluations/run', json={'suite_path': 'suite.jsonl', 'model_ref': 'tiny.pt', 'output_dir': 'evaluation'}).json()
        assert started['ok'], started
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            job = client.get('/api/jobs/' + started['job_id']).json()
            if job['done']:
                break
            time.sleep(.05)
        assert job['done'] and job['code'] == 0, job
        loaded = client.get('/api/evaluations/report', params={'path': 'evaluation/report.json'}).json()
        assert loaded['ok'] and loaded['report']['summary']['total'] == 1
        assert loaded['report']['summary']['failed'] + loaded['report']['summary']['passed'] == 1


def test_body_deadline_and_concurrent_http_budget():
    import asyncio

    from nanoforge.serving import SecurityMiddleware, ServingPolicy
    async def scenario():
        async def app(scope, receive, send):
            await receive()
        middleware = SecurityMiddleware(app, ServingPolicy(request_timeout=.02, max_concurrency=1))
        responses = []
        async def send(message):
            responses.append(message)
        async def receive():
            await asyncio.sleep(.2)
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        scope = {'type': 'http', 'path': '/api/jobs', 'headers': []}
        tasks = [asyncio.create_task(middleware(scope, receive, send)) for _ in range(5)]
        await asyncio.gather(*tasks)
        statuses = [entry['status'] for entry in responses if entry['type'] == 'http.response.start']
        assert statuses.count(429) == 1 and statuses.count(408) == 4
    asyncio.run(scenario())


def test_job_deadline_terminates_worker_and_server_output_is_bounded(tmp_path):
    import sys

    from nanoforge.serving import EngineRunner, JobManager, ServingPolicy
    manager = JobManager(tmp_path, deadline_seconds=.1)
    job_id = manager.start_process([sys.executable, '-c', 'import time; time.sleep(20)'], job_type='test')
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not manager.status(job_id)['done']:
        time.sleep(.02)
    job = manager.status(job_id)
    manager.close()
    assert job['done'] and job['code'] != 0 and 'deadline' in job['error'].lower()
    runner = EngineRunner(ServingPolicy(max_output_bytes=5))
    with pytest.raises(ValueError, match='byte budget'):
        list(runner.stream(TestEngine(), 'q', 2))
