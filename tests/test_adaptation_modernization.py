from __future__ import annotations

# ruff: noqa: E402
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nanoforge.config import DataConfig, ModelConfig, NanoforgeConfig, TrainConfig
from nanoforge.export.gguf import collect_export_tensors, export_gguf, load_nanoforge_gguf
from nanoforge.model.lora import LoRALinear, merge_lora
from nanoforge.model.transformer import build_model
from nanoforge.quantization import prepare_inference_model
from nanoforge.training.distillation import distillation_loss
from nanoforge.training.dpo_trainer import DPOTrainer, PreferenceDataset
from nanoforge.training.preferences import sequence_logprob


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_config(tmp_path: Path, steps: int = 2) -> NanoforgeConfig:
    return NanoforgeConfig(
        model=ModelConfig(vocab_size=260, max_seq_len=16, d_model=16, n_layers=1, n_heads=2),
        training=TrainConfig(
            output_dir=str(tmp_path / "out"),
            max_steps=steps,
            micro_batch_size=1,
            grad_accum_steps=2,
            warmup_steps=0,
            warmup_ratio=0,
            tensorboard=False,
            gradient_checkpointing=False,
            precision="fp32",
            device="cpu",
            log_interval=1,
            save_interval=1,
            eval_interval=1,
            eval_steps=1,
        ),
        data=DataConfig(seq_len=16, tokenizer_type="byte"),
    )


def prefs(tmp_path: Path) -> Path:
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"prompt": f"Q{i}", "chosen": "yes", "rejected": "no"}) for i in range(4)
        ),
        encoding="utf-8",
    )
    return path


def test_preference_masked_padding_is_batch_invariant():
    logits = torch.zeros(1, 2, 8)
    labels = torch.tensor([[1, 2]])
    expected = sequence_logprob(logits, labels)
    padded = sequence_logprob(torch.zeros(1, 4, 8), torch.tensor([[1, 2, -100, -100]]))
    assert torch.allclose(expected, padded)


def test_distillation_masks_and_normalizes_tokens():
    student = torch.zeros(1, 2, 3, requires_grad=True)
    teacher = torch.tensor([[[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    loss = distillation_loss(student, teacher, torch.tensor([[0, -100]]), alpha=1)
    loss.backward()
    assert torch.equal(student.grad[0, 1], torch.zeros(3))
    repeated = distillation_loss(
        student.detach().repeat(1, 4, 1),
        teacher.repeat(1, 4, 1),
        torch.tensor([[0, -100]]).repeat(1, 4),
        alpha=1,
    )
    assert torch.allclose(loss.detach(), repeated)
    assert distillation_loss(student, teacher, torch.full((1, 2), -100)).item() == 0
    with pytest.raises(ValueError):
        distillation_loss(student, teacher, torch.zeros((1, 2), dtype=torch.long), temperature=0)


def test_lora_merge_preserves_output_and_does_not_mutate():
    layer = LoRALinear(4, 3, rank=2, alpha=4).eval()
    with torch.no_grad():
        layer.lora_b.fill_(0.1)
    x = torch.randn(2, 4)
    merged = merge_lora(layer)
    assert isinstance(merged, torch.nn.Linear)
    assert torch.allclose(layer(x), merged(x), atol=1e-6)
    assert isinstance(layer, LoRALinear)


def test_int8_quantizes_custom_attention_and_rejects_unavailable():
    model = build_model(
        NanoforgeConfig(
            model=ModelConfig(vocab_size=32, max_seq_len=8, d_model=16, n_layers=1, n_heads=2)
        )
    )
    quantized = prepare_inference_model(model, "int8", device="cpu")
    assert isinstance(
        quantized.backbone.blocks[0].attn.q_proj, torch.ao.nn.quantized.dynamic.Linear
    )
    assert isinstance(model.backbone.blocks[0].attn.q_proj, LoRALinear)
    assert torch.isfinite(quantized(torch.ones(1, 3, dtype=torch.long)).logits).all()
    with pytest.raises(NotImplementedError):
        prepare_inference_model(model, "int4")
    with pytest.raises(ValueError, match="CPU"):
        prepare_inference_model(model, "int8", device="cuda")


def test_export_preserves_output_and_adapter_reload(tmp_path):
    config = NanoforgeConfig(
        model=ModelConfig(
            vocab_size=260,
            max_seq_len=8,
            d_model=16,
            n_layers=1,
            n_heads=2,
            tie_embeddings=False,
            lora_rank=2,
        )
    )
    model = build_model(config).eval()
    with torch.no_grad():
        model.backbone.blocks[0].attn.q_proj.lora_b.fill_(0.1)
    tensors = collect_export_tensors(model, config)
    assert "output.weight" in tensors
    assert all("lora_" not in key for key in tensors)
    path = tmp_path / "model.gguf"
    export_gguf(model, config, path)
    restored, restored_config, tokenizer = load_nanoforge_gguf(path)
    ids = torch.tensor([[1, 5, 6]])
    assert torch.allclose(model(ids).logits, restored(ids).logits, atol=1e-6)
    assert restored_config.model.lora_rank == 0
    assert tokenizer.vocab_size == 260
    model.extra = torch.nn.Linear(16, 16)
    with pytest.raises(ValueError, match="unsupported"):
        export_gguf(model, config, tmp_path / "bad.gguf")


def test_preference_dataset_strict_validation_and_fingerprint(tmp_path):
    from nanoforge.data.tokenizer import ByteTokenizer

    path = prefs(tmp_path)
    dataset = PreferenceDataset(path, ByteTokenizer(), 16)
    assert len(dataset) == 4
    assert dataset.stats["accepted"] == 4 and dataset.fingerprint
    path.write_text('{"chosen":"yes"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        PreferenceDataset(path, ByteTokenizer(), 16)


def test_dpo_accumulation_and_exact_resume(tmp_path):
    data = prefs(tmp_path)
    cfg = tiny_config(tmp_path / "continuous", 3)
    continuous = DPOTrainer(cfg, data)
    continuous.train()
    resumed_cfg = tiny_config(tmp_path / "resumed", 3)
    resumed_cfg.training.resume_from_checkpoint = str(tmp_path / "continuous" / "out" / "step-1.pt")
    resumed = DPOTrainer(resumed_cfg, data)
    assert resumed.completed_updates == 1
    assert all(
        torch.equal(a, b)
        for a, b in zip(resumed.ref_model.parameters(), continuous.ref_model.parameters())
    )
    resumed.train()
    assert all(
        torch.equal(a, b) for a, b in zip(resumed.model.parameters(), continuous.model.parameters())
    )
    assert continuous.forward_microsteps == 6


def test_pipeline_weights_only_runs_full_second_stage(tmp_path):
    import numpy as np

    from nanoforge.config import save_config
    from nanoforge.training.pipeline import PipelineRecipe, StageSpec, run_pipeline

    packed = tmp_path / "packed"
    packed.mkdir()
    for split in ("train", "val"):
        (packed / f"{split}.bin").write_bytes(np.arange(64, dtype=np.uint16).tobytes())
        (packed / f"{split}.manifest.json").write_text(
            json.dumps(
                {
                    "dtype": "uint16",
                    "files": [f"{split}.bin"],
                    "sequence_length": 8,
                    "labels_are_shifted": False,
                }
            )
        )
    cfg = tiny_config(tmp_path, 3)
    cfg.data.train_path, cfg.data.val_path, cfg.data.seq_len = (
        str(packed / "train.bin"),
        str(packed / "val.bin"),
        8,
    )
    cfg.training.eval_interval = 99
    base = tmp_path / "base.yaml"
    save_config(cfg, base)
    run_pipeline(
        PipelineRecipe(
            str(base),
            [StageSpec("first"), StageSpec("second", {"training": {"max_steps": 2}})],
            str(tmp_path / "pipeline"),
        )
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "pipeline" / "second" / "metrics.jsonl").read_text().splitlines()
    ]
    assert sum(event["event"] == "train" for event in events) == 2


def test_import_registry_migrates_and_rejects_future(tmp_path):
    from nanoforge.imports import ImportedModel, load_registry, save_registry

    path = tmp_path / "registry.json"
    legacy = {
        "tiny": {
            "name": "tiny",
            "source": "model.gguf",
            "backend": "llama_cpp",
            "format": "gguf",
            "tokenizer": None,
        }
    }
    path.write_text(json.dumps(legacy))
    assert load_registry(path)["tiny"].source == "model.gguf"
    save_registry({"tiny": ImportedModel(**legacy["tiny"])}, path)
    assert json.loads(path.read_text())["schema_version"] == 1
    path.write_text('{"schema_version":99,"models":{}}')
    with pytest.raises(ValueError, match="schema"):
        load_registry(path)


def test_remote_import_requires_explicit_network():
    from nanoforge.imports import load_imported_engine

    with pytest.raises(ValueError, match="network"):
        load_imported_engine("example/model", allow_network=False)


def test_pretrained_lora_real_train_evaluate_reload(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from nanoforge.training.pretrained import (
        PretrainedAdaptationConfig,
        load_pretrained_adaptation,
        train_pretrained,
    )

    source = tmp_path / "base"
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "Q": 4,
        "yes": 5,
        "no": 6,
        "0": 7,
        "1": 8,
        "2": 9,
        "3": 10,
    }
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}{{ 'Q' if add_generation_prompt else '' }}"
    tokenizer.save_pretrained(source)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=2,
            eos_token_id=3,
        )
    ).save_pretrained(source)
    data = tmp_path / "sft.jsonl"
    data.write_text(
        "\n".join(json.dumps({"prompt": f"Q {i}", "response": "yes"}) for i in range(4)),
        encoding="utf-8",
    )
    config = PretrainedAdaptationConfig(
        str(source),
        str(data),
        str(tmp_path / "adapted"),
        max_steps=2,
        seq_len=16,
        batch_size=1,
        target_modules=["c_attn"],
        lora_rank=2,
    )
    report = train_pretrained(config)
    assert report["completed_updates"] == 2 and report["evaluation"]["held_out_loss"] >= 0
    model, restored_tokenizer, manifest = load_pretrained_adaptation(tmp_path / "adapted")
    assert model.config.vocab_size == len(restored_tokenizer) and manifest["method"] == "lora"
    assert report["trainable_parameters"] < report["total_parameters"]
    from nanoforge.generation.sampling import SamplingConfig
    from nanoforge.imports import load_imported_engine

    engine = load_imported_engine(str(tmp_path / "adapted"))
    first = engine.complete("Q 0", 2, SamplingConfig(deterministic=True), [], min_new_tokens=1)
    assert first == engine.complete(
        "Q 0", 2, SamplingConfig(deterministic=True), [], min_new_tokens=1
    )
    assert engine.last_usage["exact"] and engine.last_usage["completion_tokens"] <= 2
    assert isinstance("".join(engine.chat_stream([("user", "Q 0")], max_new_tokens=2)), str)
    with pytest.raises(ValueError, match="CUDA"):
        train_pretrained(
            PretrainedAdaptationConfig(
                str(source),
                str(data),
                str(tmp_path / "bad-qlora"),
                method="qlora",
                seq_len=16,
                max_steps=1,
            )
        )
    tokenizer_file = tmp_path / "adapted" / "tokenizer_config.json"
    tokenizer_file.write_text(tokenizer_file.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_pretrained_adaptation(tmp_path / "adapted")


def test_import_registry_concurrent_updates_do_not_disappear(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from nanoforge.imports import import_model, load_registry

    registry = tmp_path / "imports.json"
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                import_model, f"model-{index}.gguf", f"model-{index}", registry_file=registry
            )
            for index in range(8)
        ]
        for future in futures:
            future.result()
    assert len(load_registry(registry)) == 8
    with pytest.raises(ValueError, match="replace"):
        import_model("new.gguf", "model-0", registry_file=registry)


def test_gguf_deterministic_and_cancellation_contract():
    import threading

    from nanoforge.imports import LlamaCppEngine

    engine = object.__new__(LlamaCppEngine)
    arguments = {}

    def generate(prompt, **kwargs):
        arguments.update(kwargs)
        yield {"choices": [{"text": "first"}]}
        yield {"choices": [{"text": "second"}]}

    engine.model, engine._lock = generate, threading.Lock()
    from nanoforge.generation.sampling import SamplingConfig

    assert engine.complete("test", sampling=SamplingConfig(deterministic=True)) == "firstsecond"
    assert arguments["temperature"] == 0
    assert engine.complete("test", interrupt=lambda: True) == ""
    with pytest.raises(ValueError, match="unsupported"):
        engine.complete("test", arbitrary_setting=True)


def test_onnx_stream_sampling_stop_and_cancellation_contract():
    import threading
    from types import SimpleNamespace

    import numpy as np

    from nanoforge.generation.sampling import SamplingConfig
    from nanoforge.imports import OnnxRuntimeEngine

    class Tokenizer:
        eos_token_id = 7

        def encode(self, prompt):
            return [0]

        def decode(self, tokens, **kwargs):
            return "".join({1: "A", 2: "B", 3: "C"}[token] for token in tokens)

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="input_ids")]

        def run(self, names, feed):
            length = feed["input_ids"].shape[1]
            logits = np.full((1, length, 8), -100, dtype=np.float32)
            logits[:, -1, min(length, 3)] = 100
            return [logits]

    engine = object.__new__(OnnxRuntimeEngine)
    engine.tokenizer, engine.session, engine.np, engine._lock = (
        Tokenizer(),
        Session(),
        np,
        threading.Lock(),
    )
    engine._native_tokenizer, engine.context_length = False, None
    assert engine.complete("Q", 3, SamplingConfig(deterministic=True), stop=["BC"]) == "A"
    assert engine.complete("Q", 3, interrupt=lambda: True) == ""


def test_real_onnx_export_dynamic_lengths_and_native_tokenizer(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    from nanoforge.export.onnx import export_onnx
    from nanoforge.imports import load_imported_engine

    config = NanoforgeConfig(
        model=ModelConfig(vocab_size=260, max_seq_len=16, d_model=16, n_layers=1, n_heads=2),
        data=DataConfig(tokenizer_type="byte"),
    )
    model = build_model(config).eval()
    path = tmp_path / "native.onnx"
    export_onnx(model, path, config=config)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for length in (2, 5):
        ids = np.full((1, length), 5, dtype=np.int64)
        with torch.inference_mode():
            expected = model(torch.from_numpy(ids)).logits.numpy()
        actual = session.run(None, {"input_ids": ids})[0]
        assert np.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    engine = load_imported_engine(str(path))
    assert engine.tokenizer.vocab_size == 260
    assert isinstance(engine.complete("A", max_new_tokens=2), str)


def test_distillation_can_load_a_larger_native_teacher(tmp_path):
    import numpy as np

    from nanoforge.training.checkpoint import save_checkpoint
    from nanoforge.training.distillation_trainer import DistillationTrainer

    cfg = tiny_config(tmp_path, 1)
    cfg.data.seq_len = 8
    packed = tmp_path / "packed"
    packed.mkdir()
    for split in ("train", "val"):
        (packed / f"{split}.bin").write_bytes(np.arange(64, dtype=np.uint16).tobytes())
        (packed / f"{split}.manifest.json").write_text(
            json.dumps(
                {
                    "dtype": "uint16",
                    "files": [f"{split}.bin"],
                    "sequence_length": 8,
                    "labels_are_shifted": False,
                }
            )
        )
    cfg.data.train_path, cfg.data.val_path = str(packed / "train.bin"), str(packed / "val.bin")
    teacher_config = NanoforgeConfig(
        model=ModelConfig(vocab_size=260, max_seq_len=16, d_model=32, n_layers=2, n_heads=2),
        data=DataConfig(tokenizer_type="byte"),
    )
    teacher = build_model(teacher_config)
    checkpoint = tmp_path / "teacher.pt"
    save_checkpoint(checkpoint, teacher, None, teacher_config, 0)
    cfg.training.teacher_checkpoint = str(checkpoint)
    trainer = DistillationTrainer(cfg)
    assert trainer.teacher.config.n_layers == 2
    trainer.train()
    assert (tmp_path / "out" / "last.pt").exists()


def test_registry_workspace_paths_preserve_backend_and_tokenizer(tmp_path):
    from nanoforge.imports import ImportedModel, resolve_imported_model, save_registry

    registry = tmp_path / "workspace" / "models" / "imports.json"
    save_registry(
        {
            "tiny": ImportedModel(
                "tiny", "models/model.onnx", "onnxruntime", "onnx", "models/tokenizer"
            )
        },
        registry,
    )
    entry = resolve_imported_model("tiny", registry_file=registry)
    assert entry.backend == "onnxruntime"
    assert entry.source == str(registry.parent.parent / "models/model.onnx")
    assert entry.tokenizer == str(registry.parent.parent / "models/tokenizer")
