from __future__ import annotations

import contextlib
import json
import os
import struct
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from nanoforge.artifacts import atomic_write_json
from nanoforge.generation.sampling import SamplingConfig

REGISTRY_SCHEMA_VERSION = 1
_registry_mutex = threading.RLock()


@dataclass
class ImportedModel:
    name: str
    source: str
    backend: str
    format: str
    tokenizer: str | None = None

    def __post_init__(self):
        if (
            not self.name
            or not self.source
            or self.backend
            not in {"llama_cpp", "transformers", "onnxruntime", "safetensors", "nanoforge_gguf"}
        ):
            raise ValueError("invalid imported model entry")


def registry_path() -> Path:
    override = os.environ.get("NANOFORGE_MODEL_REGISTRY")
    return Path(override) if override else Path("models/imports.json")


@contextlib.contextmanager
def _registry_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + 5
    with _registry_mutex:
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "import registry is locked; inspect the lock owner before recovery"
                    )
                time.sleep(0.02)
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            lock.unlink(missing_ok=True)


def load_registry(path: str | Path | None = None) -> dict[str, ImportedModel]:
    path = Path(path) if path is not None else registry_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model registry must be a JSON object")
    if "schema_version" in raw:
        if (
            type(raw["schema_version"]) is not int
            or raw["schema_version"] != REGISTRY_SCHEMA_VERSION
            or not isinstance(raw.get("models"), dict)
        ):
            raise ValueError("unsupported model registry schema")
        raw = raw["models"]
    result = {name: ImportedModel(**payload) for name, payload in raw.items()}
    if any(name != entry.name for name, entry in result.items()):
        raise ValueError("registry key/model name mismatch")
    return result


def _write_registry(models, path):
    atomic_write_json(
        path,
        {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "models": {name: asdict(model) for name, model in sorted(models.items())},
        },
    )


def save_registry(models: dict[str, ImportedModel], path: str | Path | None = None) -> None:
    target = Path(path) if path is not None else registry_path()
    with _registry_lock(target):
        _write_registry(models, target)


def import_model(
    source: str, name: str, *, tokenizer=None, backend=None, replace=False, registry_file=None
) -> ImportedModel:
    fmt, detected = detect_model_format(source)
    entry = ImportedModel(name, source, backend or detected, fmt, tokenizer)
    target = Path(registry_file) if registry_file else registry_path()
    with _registry_lock(target):
        models = load_registry(target)
        if name in models and not replace:
            raise ValueError(f"model alias {name!r} already exists; explicit replace is required")
        models[name] = entry
        _write_registry(models, target)
    return entry


def resolve_imported_model(name_or_source: str, *, registry_file=None) -> ImportedModel:
    path = Path(name_or_source)
    if (
        path.exists()
        or "/" in name_or_source
        or "\\" in name_or_source
        or path.suffix.lower() in {".gguf", ".onnx", ".safetensors"}
    ):
        fmt, backend = detect_model_format(name_or_source)
        return ImportedModel(path.stem or name_or_source, name_or_source, backend, fmt)
    models = load_registry(registry_file)
    if name_or_source in models:
        entry = models[name_or_source]
        if registry_file is not None:
            base = Path(registry_file).resolve().parent
            if base.name == "models":
                base = base.parent
            source = (
                str(base / entry.source) if not Path(entry.source).is_absolute() else entry.source
            )
            tokenizer = (
                str(base / entry.tokenizer)
                if entry.tokenizer and not Path(entry.tokenizer).is_absolute()
                else entry.tokenizer
            )
            return ImportedModel(entry.name, source, entry.backend, entry.format, tokenizer)
        return entry
    return ImportedModel(name_or_source, name_or_source, "transformers", "huggingface")


def detect_model_format(source: str) -> tuple[str, str]:
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".gguf":
        if path.exists():
            from nanoforge.export.gguf import read_gguf_metadata

            try:
                if read_gguf_metadata(path).get("general.architecture") == "nanoforge":
                    return "gguf", "nanoforge_gguf"
            except (ValueError, OSError, UnicodeError, struct.error):
                pass
        return "gguf", "llama_cpp"
    if suffix == ".onnx":
        return "onnx", "onnxruntime"
    if suffix == ".safetensors":
        return "safetensors", "transformers" if (
            path.parent / "config.json"
        ).exists() else "safetensors"
    if path.is_dir() and (path / "config.json").exists():
        return "huggingface", "transformers"
    if path.is_dir() and (path / "adaptation.manifest.json").exists():
        return "adapter", "transformers"
    if "/" in source and not path.exists():
        return "huggingface", "transformers"
    return "unknown", "transformers"


def load_imported_engine(
    name_or_source: str, *, device="auto", allow_network=False, registry_path=None, **options
):
    entry = resolve_imported_model(name_or_source, registry_file=registry_path)
    if entry.backend != "llama_cpp" and options:
        raise ValueError(f"unsupported loader options for {entry.backend}: {sorted(options)}")
    source = Path(entry.source)
    if not source.exists() and not allow_network:
        raise ValueError("model source is not local; network access requires allow_network=True")
    if entry.backend == "nanoforge_gguf":
        from nanoforge.export.gguf import load_nanoforge_gguf
        from nanoforge.generation.engine import GenerationEngine

        model, config, tokenizer = load_nanoforge_gguf(entry.source)
        return GenerationEngine(model, tokenizer, device=device, config=config)
    if entry.backend == "llama_cpp":
        return LlamaCppEngine(entry, device=device, **options)
    if entry.backend == "transformers":
        return TransformersEngine(entry, device=device, allow_network=allow_network)
    if entry.backend == "onnxruntime":
        if device not in {"auto", "cpu"}:
            raise ValueError("this ONNX adapter supports CPU inference only")
        return OnnxRuntimeEngine(entry, allow_network=allow_network)
    raise RuntimeError("standalone safetensors require a model config and tokenizer directory")


def _check_controls(options):
    allowed = {"interrupt", "stop", "stop_sequences"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unsupported imported-engine controls: {sorted(unknown)}")
    if options.get("interrupt") is not None and not callable(options["interrupt"]):
        raise ValueError("interrupt must be callable")
    _stops(options)


def _stops(options):
    stops = options.get("stop") or options.get("stop_sequences") or []
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, (list, tuple)) or any(
        not isinstance(stop, str) or not stop for stop in stops
    ):
        raise ValueError("stop sequences must be nonempty strings")
    return stops


def _visible_text(decoded, stops):
    positions = [decoded.find(stop) for stop in stops if stop in decoded]
    if positions:
        return decoded[: min(positions)], True
    held = max(
        [
            length
            for stop in stops
            for length in range(1, len(stop))
            if decoded.endswith(stop[:length])
        ]
        or [0]
    )
    return (decoded[:-held] if held else decoded), False


def _chat_format(tokenizer, messages):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "selected tokenizer has no chat template; choose an explicit project chat format"
        )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _native_controls(stop_tokens, min_new_tokens, constraint, options):
    if constraint is not None:
        raise NotImplementedError("grammar constraints are unsupported by this imported engine")
    if min_new_tokens < 0:
        raise ValueError("minimum token budget must be nonnegative")
    if stop_tokens is not None:
        if options.get("stop") or options.get("stop_sequences"):
            raise ValueError("choose stop_tokens or stop, not both")
        options["stop"] = stop_tokens
    _check_controls(options)


class _ImportedChat:
    def chat_stream(
        self, messages, max_new_tokens=256, sampling=None, system=None, min_new_tokens=1, **options
    ):
        conversation = []
        if system:
            conversation.append({"role": "system", "content": system})
        for message in messages:
            item = (
                message
                if isinstance(message, dict)
                else {"role": message[0], "content": message[1]}
            )
            if item.get("role") not in {"system", "user", "assistant", "tool"} or not isinstance(
                item.get("content"), str
            ):
                raise ValueError("chat messages require a known role and string content")
            conversation.append(item)
        yield from self.stream(
            self.format_chat(conversation),
            max_new_tokens,
            sampling or SamplingConfig(mode="chat"),
            min_new_tokens=min_new_tokens,
            **options,
        )


class LlamaCppEngine(_ImportedChat):
    capabilities = {
        "streaming": True,
        "cancellation": "cooperative_between_tokens",
        "sampling": True,
        "chat_template": False,
        "network": False,
    }

    def __init__(self, entry, *, device="auto", context_length=2048, threads=None, gpu_layers=None):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Install latticekin[gguf] to run external GGUF models") from exc
        if context_length <= 0 or (threads is not None and threads <= 0):
            raise ValueError("context/thread budgets must be positive")
        self.entry = entry
        layers = (0 if device in {"cpu", "auto"} else -1) if gpu_layers is None else gpu_layers
        self.model = Llama(
            model_path=entry.source,
            n_ctx=context_length,
            n_threads=threads or max(1, os.cpu_count() or 1),
            n_gpu_layers=layers,
        )
        self._lock = threading.Lock()

    def format_chat(self, messages):
        template = getattr(self.model, "metadata", {}).get("tokenizer.chat_template")
        if not template:
            raise ValueError("GGUF has no chat template; select an explicit project format")
        try:
            from transformers.utils.chat_template_utils import _compile_jinja_template
        except ImportError as error:
            raise RuntimeError(
                "GGUF chat template rendering requires latticekin[adaptation]"
            ) from error
        bos = self.model.detokenize([self.model.token_bos()], special=True).decode("utf-8")
        eos = self.model.detokenize([self.model.token_eos()], special=True).decode("utf-8")
        return _compile_jinja_template(template).render(
            messages=messages, add_generation_prompt=True, bos_token=bos, eos_token=eos
        )

    def stream(
        self,
        prompt,
        max_new_tokens=256,
        sampling=None,
        stop_tokens=None,
        min_new_tokens=0,
        constraint=None,
        **options,
    ):
        _native_controls(stop_tokens, min_new_tokens, constraint, options)
        if max_new_tokens < 0:
            raise ValueError("token budget must be nonnegative")
        if max_new_tokens == 0:
            return
        sampling = sampling or SamplingConfig()
        extra = {}
        if min_new_tokens:
            from llama_cpp import LogitsProcessorList

            prompt_length = len(self.model.tokenize(prompt.encode("utf-8"), add_bos=True))
            eos = self.model.token_eos()

            def minimum_length(input_ids, logits):
                if len(input_ids) < prompt_length + min_new_tokens:
                    logits[eos] = float("-inf")
                return logits

            extra["logits_processor"] = LogitsProcessorList([minimum_length])
        self.last_usage = {"exact": False, "prompt_tokens": None, "completion_tokens": None}
        if (
            sampling.mirostat
            or sampling.entropy_target is not None
            or sampling.adaptive_temperature
            or sampling.no_repeat_ngram_size
        ):
            raise ValueError(
                "selected GGUF adapter does not support these advanced sampling controls"
            )
        interrupt = options.get("interrupt")
        with self._lock:
            generator = self.model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=0.0 if sampling.deterministic else max(sampling.temperature, 0),
                top_k=sampling.top_k or 0,
                top_p=sampling.top_p or 1.0,
                min_p=sampling.min_p or 0,
                repeat_penalty=sampling.repetition_penalty,
                frequency_penalty=sampling.frequency_penalty,
                presence_penalty=sampling.presence_penalty,
                stop=_stops(options),
                stream=True,
                **extra,
            )
            try:
                for chunk in generator:
                    if interrupt and interrupt():
                        break
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    if text:
                        yield text
            finally:
                close = getattr(generator, "close", None)
                if close:
                    close()

    def complete(
        self, prompt, max_new_tokens=256, sampling=None, stop_tokens=None, constraint=None, **kwargs
    ):
        return "".join(
            self.stream(
                prompt, max_new_tokens, sampling, stop_tokens, constraint=constraint, **kwargs
            )
        )


class TransformersEngine(_ImportedChat):
    capabilities = {
        "streaming": True,
        "cancellation": "cooperative_between_tokens",
        "sampling": True,
        "chat_template": True,
        "network": "explicit_opt_in",
    }

    def __init__(self, entry, *, device="auto", allow_network=False):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install latticekin[adaptation] for Hugging Face models") from exc
        from nanoforge.training.utils import resolve_device

        self.torch, self.entry = torch, entry
        source = str(Path(entry.source).parent) if entry.format == "safetensors" else entry.source
        if (Path(source) / "adaptation.manifest.json").exists():
            from nanoforge.training.pretrained import load_pretrained_adaptation

            if entry.tokenizer:
                raise ValueError("adaptation bundles use their saved tokenizer identity")
            self.device = resolve_device(device)
            self.model, self.tokenizer, _ = load_pretrained_adaptation(
                source, device=str(self.device), allow_network=allow_network
            )
            self._lock = threading.Lock()
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            entry.tokenizer or source, local_files_only=not allow_network, trust_remote_code=False
        )
        self.device = resolve_device(device)
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                source,
                torch_dtype=dtype,
                local_files_only=not allow_network,
                trust_remote_code=False,
            )
            .to(self.device)
            .eval()
        )
        if self.model.get_input_embeddings().num_embeddings != len(self.tokenizer):
            raise ValueError("imported model/tokenizer vocabulary mismatch")
        self._lock = threading.Lock()

    def format_chat(self, messages):
        return _chat_format(self.tokenizer, messages)

    def stream(
        self,
        prompt,
        max_new_tokens=256,
        sampling=None,
        stop_tokens=None,
        min_new_tokens=0,
        constraint=None,
        **options,
    ) -> Iterator[str]:
        _native_controls(stop_tokens, min_new_tokens, constraint, options)
        from nanoforge.generation.sampling import MirostatState, sample_next

        torch = self.torch
        sampling = sampling or SamplingConfig()
        interrupt = options.get("interrupt")
        stops = _stops(options)
        if max_new_tokens < 0:
            raise ValueError("token budget must be nonnegative")
        generated, published, cache = [], "", None
        matched, cancelled = False, False
        encoded = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        if encoded.numel() == 0:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                raise ValueError("empty prompt requires a BOS token")
            encoded = torch.tensor([[bos]], device=self.device)
        context = getattr(
            self.model.config,
            "max_position_embeddings",
            getattr(self.model.config, "n_positions", None),
        )
        if context and encoded.shape[1] + max_new_tokens > context:
            raise ValueError("prompt plus token budget exceeds model context")
        self.last_usage = {"exact": True, "prompt_tokens": encoded.shape[1], "completion_tokens": 0}
        mirostat = MirostatState(sampling.mirostat_tau) if sampling.mirostat else None
        with self._lock, torch.inference_mode():
            for _ in range(max_new_tokens):
                if interrupt and interrupt():
                    cancelled = True
                    break
                output = self.model(encoded, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                if len(generated) < min_new_tokens and self.tokenizer.eos_token_id is not None:
                    output.logits = output.logits.clone()
                    output.logits[:, -1, self.tokenizer.eos_token_id] = float("-inf")
                history = torch.tensor(generated, dtype=torch.long, device=self.device).reshape(
                    1, -1
                )
                token = sample_next(output.logits, history, sampling, mirostat)
                token_id = int(token.item())
                self.last_usage["completion_tokens"] += 1
                self.last_usage["total_tokens"] = (
                    self.last_usage["prompt_tokens"] + self.last_usage["completion_tokens"]
                )
                if (
                    self.tokenizer.eos_token_id is not None
                    and token_id == self.tokenizer.eos_token_id
                ):
                    break
                generated.append(token_id)
                encoded = token.reshape(1, 1)
                decoded = self.tokenizer.decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ).rstrip("\ufffd")
                decoded, matched = _visible_text(decoded, stops)
                if not decoded.startswith(published):
                    raise RuntimeError(
                        "tokenizer decoding is not prefix-stable; cannot safely stream"
                    )
                chunk = decoded[len(published) :]
                if chunk:
                    yield chunk
                published = decoded
                if matched:
                    break
            if not matched and not cancelled:
                decoded = self.tokenizer.decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                if decoded.startswith(published) and decoded[len(published) :]:
                    yield decoded[len(published) :]

    def complete(
        self, prompt, max_new_tokens=256, sampling=None, stop_tokens=None, constraint=None, **kwargs
    ):
        return "".join(
            self.stream(
                prompt, max_new_tokens, sampling, stop_tokens, constraint=constraint, **kwargs
            )
        )


class OnnxRuntimeEngine(_ImportedChat):
    capabilities = {
        "streaming": True,
        "cancellation": "cooperative_between_tokens",
        "sampling": True,
        "chat_template": True,
        "kv_cache": False,
        "network": "explicit_tokenizer_opt_in",
    }

    def __init__(self, entry, *, allow_network=False):
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install latticekin[export,adaptation] for ONNX imports") from exc
        self.entry, self.np = entry, np
        self.session = ort.InferenceSession(entry.source, providers=["CPUExecutionProvider"])
        sidecar = Path(entry.source).with_suffix(Path(entry.source).suffix + ".nanoforge.json")
        self._native_tokenizer = sidecar.exists() and entry.tokenizer is None
        self.context_length = None
        if self._native_tokenizer:
            from nanoforge.export.gguf import restore_tokenizer_payload

            bundle = json.loads(sidecar.read_text(encoding="utf-8"))
            if bundle.get("schema_version") != 1:
                raise ValueError("unsupported ONNX tokenizer bundle schema")
            self.tokenizer = restore_tokenizer_payload(bundle["tokenizer"])
            self.context_length = bundle["context_length"]
        else:
            try:
                from transformers import AutoTokenizer
            except ImportError as error:
                raise RuntimeError(
                    "Foreign ONNX tokenizers require latticekin[adaptation]"
                ) from error
            self.tokenizer = AutoTokenizer.from_pretrained(
                entry.tokenizer or str(Path(entry.source).parent),
                local_files_only=not allow_network,
                trust_remote_code=False,
            )
        self._lock = threading.Lock()

    def format_chat(self, messages):
        return _chat_format(self.tokenizer, messages)

    def stream(
        self,
        prompt,
        max_new_tokens=256,
        sampling=None,
        stop_tokens=None,
        min_new_tokens=0,
        constraint=None,
        **options,
    ):
        _native_controls(stop_tokens, min_new_tokens, constraint, options)
        import torch

        from nanoforge.generation.sampling import MirostatState, sample_next

        sampling = sampling or SamplingConfig()
        stops = _stops(options)
        if max_new_tokens < 0:
            raise ValueError("token budget must be nonnegative")
        ids = self.tokenizer.encode(prompt)
        if not ids:
            bos = self.tokenizer.bos_id if self._native_tokenizer else self.tokenizer.bos_token_id
            if bos is None:
                raise ValueError("empty ONNX prompt requires a BOS token")
            ids = [bos]
        if self.context_length and len(ids) + max_new_tokens > self.context_length:
            raise ValueError("prompt plus token budget exceeds ONNX model context")
        self.last_usage = {"exact": True, "prompt_tokens": len(ids), "completion_tokens": 0}
        generated, published = [], ""
        matched, cancelled = False, False
        names = {item.name for item in self.session.get_inputs()}
        if names - {"input_ids", "attention_mask"}:
            raise ValueError(f"unsupported ONNX input contract: {sorted(names)}")
        mirostat = MirostatState(sampling.mirostat_tau) if sampling.mirostat else None
        with self._lock:
            for _ in range(max_new_tokens):
                if options.get("interrupt") and options["interrupt"]():
                    cancelled = True
                    break
                array = self.np.asarray([ids], dtype=self.np.int64)
                feed = {"input_ids": array}
                if "attention_mask" in names:
                    feed["attention_mask"] = self.np.ones_like(array)
                logits = torch.from_numpy(self.session.run(None, feed)[0])
                eos = (
                    self.tokenizer.eos_id if self._native_tokenizer else self.tokenizer.eos_token_id
                )
                if len(generated) < min_new_tokens and eos is not None:
                    logits[:, -1, eos] = float("-inf")
                token = int(
                    sample_next(
                        logits, torch.tensor([generated], dtype=torch.long), sampling, mirostat
                    ).item()
                )
                eos = (
                    self.tokenizer.eos_id if self._native_tokenizer else self.tokenizer.eos_token_id
                )
                self.last_usage["completion_tokens"] += 1
                self.last_usage["total_tokens"] = (
                    self.last_usage["prompt_tokens"] + self.last_usage["completion_tokens"]
                )
                if token == eos:
                    break
                ids.append(token)
                generated.append(token)
                text = (
                    self.tokenizer.decode(generated)
                    if self._native_tokenizer
                    else self.tokenizer.decode(
                        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                ).rstrip("\ufffd")
                text, matched = _visible_text(text, stops)
                if not text.startswith(published):
                    raise RuntimeError("ONNX tokenizer decoding is not prefix-stable")
                if text[len(published) :]:
                    yield text[len(published) :]
                published = text
                if matched:
                    break
            if not matched and not cancelled:
                text = (
                    self.tokenizer.decode(generated)
                    if self._native_tokenizer
                    else self.tokenizer.decode(
                        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                )
                if text.startswith(published) and text[len(published) :]:
                    yield text[len(published) :]

    def complete(
        self, prompt, max_new_tokens=256, sampling=None, stop_tokens=None, constraint=None, **kwargs
    ):
        return "".join(
            self.stream(
                prompt, max_new_tokens, sampling, stop_tokens, constraint=constraint, **kwargs
            )
        )
