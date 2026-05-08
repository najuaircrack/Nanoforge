from __future__ import annotations

from dataclasses import dataclass
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
        stream: bool = False

    app = FastAPI(title="Nanoforge API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/v1/completions")
    def completions(req: RequestModel):
        sampling = SamplingConfig(
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
        if req.stream:
            def iterator() -> Iterator[str]:
                for piece in engine.stream(req.prompt, req.max_new_tokens, sampling):
                    yield piece

            return StreamingResponse(iterator(), media_type="text/plain")
        return {"text": engine.complete(req.prompt, req.max_new_tokens, sampling)}

    return app


def serve(checkpoint: str, host: str = "127.0.0.1", port: int = 8000, device: str = "auto") -> None:
    try:
        import uvicorn
    except Exception as exc:
        raise RuntimeError("Install server extras: pip install -e .[serve]") from exc
    engine = GenerationEngine.from_checkpoint(checkpoint, device=device)
    uvicorn.run(create_app(engine), host=host, port=port)
