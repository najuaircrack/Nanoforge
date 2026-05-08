from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterator

from nanoforge.generation.engine import GenerationEngine
from nanoforge.generation.sampling import SamplingConfig


@dataclass
class CompletionRequest:
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = 0.95
    repetition_penalty: float = 1.0
    stream: bool = False
    mode: str = "balanced"
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None


def create_app(engine: GenerationEngine):
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except Exception as exc:
        raise RuntimeError("Install server extras: pip install -e .[serve]") from exc

    class RequestModel(BaseModel):
        prompt: str
        max_new_tokens: int = 256
        temperature: float = 0.8
        top_k: int | None = 50
        top_p: float | None = 0.95
        repetition_penalty: float = 1.0
        frequency_penalty: float = 0.0
        presence_penalty: float = 0.0
        mode: str = "balanced"
        stop: list[str] | None = None
        stream: bool = False

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequestModel(BaseModel):
        messages: list[ChatMessage]
        max_tokens: int = 256
        temperature: float = 0.7
        top_p: float | None = 0.9
        stop: list[str] | None = None
        stream: bool = False

    app = FastAPI(title="Nanoforge API", version="0.1.0")
    started = time.time()
    counters = {"requests": 0, "tokens": 0}

    @app.get("/health")
    def health():
        return {"ok": True, "uptime_seconds": time.time() - started}

    @app.get("/metrics")
    def metrics():
        return {
            "nanoforge_requests_total": counters["requests"],
            "nanoforge_generated_tokens_total": counters["tokens"],
            "nanoforge_uptime_seconds": time.time() - started,
        }

    @app.post("/v1/completions")
    def completions(req: RequestModel):
        sampling = SamplingConfig(
            mode=req.mode,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
        )
        counters["requests"] += 1
        if req.stream:
            def iterator() -> Iterator[str]:
                for piece in engine.stream(req.prompt, req.max_new_tokens, sampling, stop_tokens=req.stop):
                    counters["tokens"] += 1
                    yield piece

            return StreamingResponse(iterator(), media_type="text/plain")
        text = engine.complete(req.prompt, req.max_new_tokens, sampling, stop_tokens=req.stop)
        return {"text": text}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequestModel):
        prompt = "\n".join(f"<|{msg.role}|>\n{msg.content}" for msg in req.messages) + "\n<|assistant|>\n"
        sampling = SamplingConfig(mode="chat", temperature=req.temperature, top_p=req.top_p)
        counters["requests"] += 1
        if req.stream:
            def iterator() -> Iterator[str]:
                for piece in engine.stream(
                    prompt,
                    req.max_tokens,
                    sampling,
                    stop_tokens=req.stop or ["<|user|>", "<|system|>", "<|endoftext|>"],
                ):
                    counters["tokens"] += 1
                    yield piece

            return StreamingResponse(iterator(), media_type="text/plain")
        text = engine.complete(
            prompt,
            req.max_tokens,
            sampling,
            stop_tokens=req.stop or ["<|user|>", "<|system|>", "<|endoftext|>"],
        )
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    return app


def serve(checkpoint: str, host: str = "127.0.0.1", port: int = 8000, device: str = "auto") -> None:
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Install server extras: pip install -e .[serve]") from exc
    engine = GenerationEngine.from_checkpoint(checkpoint, device=device)
    uvicorn.run(create_app(engine), host=host, port=port)
