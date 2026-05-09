from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from nanoforge.generation.sampling import SamplingConfig


@dataclass
class ImportedModel:
    name: str
    source: str
    backend: str
    format: str
    tokenizer: str | None = None


def registry_path() -> Path:
    override = os.environ.get("NANOFORGE_MODEL_REGISTRY")
    return Path(override) if override else Path("models/imports.json")


def load_registry(path: str | Path | None = None) -> dict[str, ImportedModel]:
    path = Path(path) if path is not None else registry_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: ImportedModel(**payload) for name, payload in raw.items()}


def save_registry(models: dict[str, ImportedModel], path: str | Path | None = None) -> None:
    path = Path(path) if path is not None else registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({name: asdict(model) for name, model in models.items()}, indent=2), encoding="utf-8")


def import_model(source: str, name: str, *, tokenizer: str | None = None, backend: str | None = None) -> ImportedModel:
    fmt, resolved_backend = detect_model_format(source)
    entry = ImportedModel(name=name, source=source, backend=backend or resolved_backend, format=fmt, tokenizer=tokenizer)
    models = load_registry()
    models[name] = entry
    save_registry(models)
    return entry


def resolve_imported_model(name_or_source: str) -> ImportedModel:
    models = load_registry()
    if name_or_source in models:
        return models[name_or_source]
    fmt, backend = detect_model_format(name_or_source)
    return ImportedModel(name=Path(name_or_source).stem or name_or_source, source=name_or_source, backend=backend, format=fmt)


def detect_model_format(source: str) -> tuple[str, str]:
    path = Path(source)
    suffix = path.suffix.lower()
    if path.exists():
        if suffix == ".gguf":
            return "gguf", "llama_cpp"
        if suffix == ".onnx":
            return "onnx", "onnxruntime"
        if suffix == ".safetensors":
            if (path.parent / "config.json").exists():
                return "safetensors", "transformers"
            return "safetensors", "safetensors"
        if path.is_dir() and (path / "config.json").exists():
            return "huggingface", "transformers"
    if "/" in source and not path.exists():
        return "huggingface", "transformers"
    return "unknown", "transformers"


def load_imported_engine(name_or_source: str, *, device: str = "auto"):
    entry = resolve_imported_model(name_or_source)
    if entry.backend == "llama_cpp":
        return LlamaCppEngine(entry)
    if entry.backend == "transformers":
        return TransformersEngine(entry, device=device)
    if entry.backend == "onnxruntime":
        return OnnxRuntimeEngine(entry)
    if entry.backend == "safetensors":
        return SafeTensorsEngine(entry)
    raise ValueError(f"Unsupported imported model backend '{entry.backend}'.")


class LlamaCppEngine:
    def __init__(self, entry: ImportedModel):
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError("Install llama-cpp-python to run GGUF models.") from exc
        self.entry = entry
        self.model = Llama(model_path=entry.source, n_ctx=2048, n_threads=max(1, os.cpu_count() or 1))

    def stream(self, prompt: str, max_new_tokens: int = 256, sampling: SamplingConfig | None = None, **_) -> Iterator[str]:
        sampling = sampling or SamplingConfig()
        out = self.model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=max(sampling.temperature, 0.0),
            top_k=sampling.top_k or 0,
            top_p=sampling.top_p or 1.0,
            repeat_penalty=sampling.repetition_penalty,
            stream=True,
        )
        for chunk in out:
            text = chunk.get("choices", [{}])[0].get("text", "")
            if text:
                yield text

    def complete(self, prompt: str, max_new_tokens: int = 256, sampling: SamplingConfig | None = None, **kwargs) -> str:
        return "".join(self.stream(prompt, max_new_tokens, sampling, **kwargs))


class TransformersEngine:
    def __init__(self, entry: ImportedModel, *, device: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("Install transformers and torch to run HuggingFace/SafeTensors models.") from exc
        self.torch = torch
        self.entry = entry
        source = str(Path(entry.source).parent) if entry.format == "safetensors" else entry.source
        self.tokenizer = AutoTokenizer.from_pretrained(entry.tokenizer or source)
        self.model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32).eval()
        self.device = torch.device("cpu" if device == "auto" else device)
        self.model.to(self.device)

    def stream(self, prompt: str, max_new_tokens: int = 256, sampling: SamplingConfig | None = None, **_) -> Iterator[str]:
        yield self.complete(prompt, max_new_tokens=max_new_tokens, sampling=sampling)

    def complete(self, prompt: str, max_new_tokens: int = 256, sampling: SamplingConfig | None = None, **_) -> str:
        sampling = sampling or SamplingConfig()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=not sampling.deterministic and sampling.temperature > 0,
                temperature=max(sampling.temperature, 1e-5),
                top_k=sampling.top_k or 0,
                top_p=sampling.top_p or 1.0,
                repetition_penalty=sampling.repetition_penalty,
                no_repeat_ngram_size=sampling.no_repeat_ngram_size,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


class OnnxRuntimeEngine:
    def __init__(self, entry: ImportedModel):
        try:
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError("Install onnxruntime and transformers to run ONNX imports.") from exc
        self.entry = entry
        self.np = np
        self.session = ort.InferenceSession(entry.source, providers=["CPUExecutionProvider"])
        tokenizer_source = entry.tokenizer or str(Path(entry.source).parent)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    def stream(self, prompt: str, max_new_tokens: int = 256, sampling: SamplingConfig | None = None, **_) -> Iterator[str]:
        yield self.complete(prompt, max_new_tokens=max_new_tokens, sampling=sampling)

    def complete(self, prompt: str, max_new_tokens: int = 256, sampling: SamplingConfig | None = None, **kwargs) -> str:
        sampling = sampling or SamplingConfig(deterministic=True)
        ids = self.tokenizer.encode(prompt, return_tensors=None)
        generated: list[int] = []
        input_names = {item.name for item in self.session.get_inputs()}
        for _ in range(max_new_tokens):
            arr = self.np.asarray([ids], dtype=self.np.int64)
            feed = {}
            if "input_ids" in input_names:
                feed["input_ids"] = arr
            else:
                feed[next(iter(input_names))] = arr
            if "attention_mask" in input_names:
                feed["attention_mask"] = self.np.ones_like(arr, dtype=self.np.int64)
            logits = self.session.run(None, feed)[0]
            next_id = int(logits[:, -1, :].argmax(axis=-1)[0])
            if self.tokenizer.eos_token_id is not None and next_id == self.tokenizer.eos_token_id:
                break
            ids.append(next_id)
            generated.append(next_id)
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class SafeTensorsEngine:
    def __init__(self, entry: ImportedModel):
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError("Install safetensors to inspect standalone SafeTensors files.") from exc
        self.entry = entry
        self.safe_open = safe_open

    def stream(self, prompt: str, max_new_tokens: int = 256, sampling: SamplingConfig | None = None, **_) -> Iterator[str]:
        raise RuntimeError("Standalone .safetensors files need a HuggingFace config/tokenizer directory to generate text.")

    def complete(self, prompt: str, max_new_tokens: int = 256, sampling: SamplingConfig | None = None, **kwargs) -> str:
        return "".join(self.stream(prompt, max_new_tokens, sampling, **kwargs))
