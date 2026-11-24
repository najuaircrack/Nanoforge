"""Bounded local inference with OpenAI completion/chat wire envelopes."""
from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from nanoforge.generation.engine import GenerationEngine
from nanoforge.generation.sampling import SamplingConfig
from nanoforge.serving import (
    EngineRunner,
    SecurityMiddleware,
    ServingPolicy,
    WorkspacePaths,
    count_tokens,
)

try:
    from pydantic import AliasChoices, BaseModel, ConfigDict, Field
except ImportError:
    BaseModel = object
    ConfigDict = dict
    def AliasChoices(*values):
        return values
    def Field(default=None, **kwargs):
        return default


class RequestModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = 'latticekin'
    prompt: str = Field(max_length=1024 * 1024)
    max_new_tokens: int = Field(256, ge=1, le=4096, validation_alias=AliasChoices('max_new_tokens', 'max_tokens'))
    temperature: float = Field(.8, ge=0, le=2)
    top_k: int | None = Field(50, ge=0)
    top_p: float | None = Field(.95, gt=0, le=1)
    repetition_penalty: float = Field(1.0, gt=0)
    frequency_penalty: float = Field(0.0, ge=-2, le=2)
    presence_penalty: float = Field(0.0, ge=-2, le=2)
    mode: Literal['balanced', 'chat', 'creative', 'coding', 'deterministic', 'low_memory', 'high_quality'] = 'balanced'
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = Field(1, ge=1, le=1)
    user: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['system', 'user', 'assistant', 'tool']
    content: str = Field(max_length=1024 * 1024)
    name: str | None = None
    tool_call_id: str | None = None


class ChatRequestModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = 'latticekin'
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    max_tokens: int = Field(256, ge=1, le=4096, validation_alias=AliasChoices('max_tokens', 'max_completion_tokens', 'max_new_tokens'))
    temperature: float = Field(.7, ge=0, le=2)
    top_p: float | None = Field(.9, gt=0, le=1)
    frequency_penalty: float = Field(0.0, ge=-2, le=2)
    presence_penalty: float = Field(0.0, ge=-2, le=2)
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = Field(1, ge=1, le=1)
    user: str | None = None


def create_app(engine: GenerationEngine, *, policy: ServingPolicy | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError('Install server extras: pip install -e .[serve]') from exc
    policy = policy or ServingPolicy()
    runner = EngineRunner(policy)

    @asynccontextmanager
    async def lifespan(app):
        yield
        runner.close()

    app = FastAPI(title='Latticekin API', version='0.1.0', lifespan=lifespan)
    app.add_middleware(SecurityMiddleware, policy=policy)
    app.state.engine_runner = runner
    started = time.monotonic()
    counters = {'requests': 0, 'tokens': 0, 'usage_unavailable': 0}
    counter_lock = threading.Lock()
    model_id = getattr(getattr(engine, 'entry', None), 'name', 'latticekin')

    @app.get('/health')
    def health():
        return {'ok': True, 'uptime_seconds': time.monotonic() - started}

    @app.get('/metrics')
    def metrics():
        with counter_lock:
            return {'nanoforge_requests_total': counters['requests'], 'nanoforge_generated_tokens_total': counters['tokens'], 'usage_unavailable_requests': counters['usage_unavailable'], 'active_requests': runner.active, 'nanoforge_uptime_seconds': time.monotonic() - started}

    @app.get('/v1/models')
    def models():
        return {'object': 'list', 'data': [{'id': model_id, 'object': 'model', 'created': int(time.time()), 'owned_by': 'local'}]}

    def respond(prompt, limit, sampling, stop, stream, model_name, chat):
        if model_name not in {model_id, 'latticekin', 'nanoforge'}:
            raise HTTPException(404, 'Requested model is not served by this instance.')
        if limit > policy.max_new_tokens:
            raise HTTPException(422, 'max_tokens exceeds configured server policy.')
        with counter_lock:
            counters['requests'] += 1
        engine_usage = {}
        request_id, created = ('chatcmpl-' if chat else 'cmpl-') + uuid.uuid4().hex, int(time.time())
        stops = [stop] if isinstance(stop, str) else stop
        def usage(text):
            completion = engine_usage['generated_tokens'] if 'generated_tokens' in engine_usage else count_tokens(engine, text)
            prompt_count = engine_usage['prompt_tokens'] if 'prompt_tokens' in engine_usage else count_tokens(engine, prompt)
            with counter_lock:
                if completion is not None:
                    counters['tokens'] += completion
                else:
                    counters['usage_unavailable'] += 1
            return {'prompt_tokens': prompt_count, 'completion_tokens': completion, 'total_tokens': prompt_count + completion if prompt_count is not None and completion is not None else None}
        def envelope(choices, object_type):
            return {'id': request_id, 'object': object_type, 'created': created, 'model': model_name, 'choices': choices}
        if stream:
            def iterator():
                pieces = []
                try:
                    if chat:
                        yield 'data: ' + json.dumps(envelope([{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}], 'chat.completion.chunk')) + '\n\n'
                    for piece in runner.stream(engine, prompt, limit, sampling, stop_tokens=stops, usage=engine_usage):
                        pieces.append(piece)
                        choice = {'index': 0, 'finish_reason': None}
                        choice['delta' if chat else 'text'] = {'content': piece} if chat else piece
                        yield 'data: ' + json.dumps(envelope([choice], 'chat.completion.chunk' if chat else 'text_completion')) + '\n\n'
                    final = envelope([{'index': 0, 'finish_reason': 'stop', **({'delta': {}} if chat else {'text': ''})}], 'chat.completion.chunk' if chat else 'text_completion')
                    final['usage'] = usage(''.join(pieces))
                    final['usage_measurement'] = engine_usage.get('counting', 'visible_text_retokenized')
                    yield 'data: ' + json.dumps(final) + '\n\n'
                except Exception as exc:
                    yield 'data: ' + json.dumps({'error': {'message': str(exc), 'type': type(exc).__name__}}) + '\n\n'
                yield 'data: [DONE]\n\n'
            return StreamingResponse(iterator(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})
        try:
            text = ''.join(runner.stream(engine, prompt, limit, sampling, stop_tokens=stops, usage=engine_usage))
        except TimeoutError as exc:
            raise HTTPException(504, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(429 if 'busy' in str(exc) else 500, str(exc)) from exc
        choice = {'index': 0, 'finish_reason': 'stop', **({'message': {'role': 'assistant', 'content': text}} if chat else {'text': text, 'logprobs': None})}
        result = envelope([choice], 'chat.completion' if chat else 'text_completion')
        result.update(usage=usage(text), usage_measurement=engine_usage.get('counting', 'visible_text_retokenized'), text=text)
        return result

    @app.post('/v1/completions')
    def completions(req: RequestModel):
        sampling = SamplingConfig(mode=req.mode, temperature=req.temperature, top_k=req.top_k, top_p=req.top_p, repetition_penalty=req.repetition_penalty, frequency_penalty=req.frequency_penalty, presence_penalty=req.presence_penalty)
        return respond(req.prompt, req.max_new_tokens, sampling, req.stop, req.stream, req.model, False)

    @app.post('/v1/chat/completions')
    def chat_completions(req: ChatRequestModel):
        messages = [(message.role, message.content) for message in req.messages]
        render = getattr(engine, 'build_chat_prompt', GenerationEngine.build_chat_prompt)
        prompt = render(messages)
        sampling = SamplingConfig(mode='chat', temperature=req.temperature, top_p=req.top_p, frequency_penalty=req.frequency_penalty, presence_penalty=req.presence_penalty)
        return respond(prompt, req.max_tokens, sampling, req.stop, req.stream, req.model, True)

    return app


def serve(checkpoint: str, host: str = '127.0.0.1', port: int = 8000, device: str = 'auto', *, api_key: str | None = None, workspace_root=None, max_request_bytes=2 * 1024 * 1024, max_concurrency=2, request_timeout=120.0, allowed_origins=None) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError('Install server extras: pip install -e .[serve]') from exc
    policy = ServingPolicy(host=host, api_key=api_key, max_request_bytes=max_request_bytes, max_concurrency=max_concurrency, request_timeout=request_timeout, allowed_origins=tuple(allowed_origins or ()))
    source = WorkspacePaths(workspace_root).resolve(checkpoint) if workspace_root else checkpoint
    engine = GenerationEngine.from_checkpoint(source, device=device)
    uvicorn.run(create_app(engine, policy=policy), host=host, port=port)
