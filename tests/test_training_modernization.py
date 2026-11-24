from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

from nanoforge.config import DataConfig, ModelConfig, NanoforgeConfig, TrainConfig
from nanoforge.training.checkpoint import load_checkpoint, migrate_checkpoint, save_checkpoint
from nanoforge.training.optimizers import Adafactor
from nanoforge.training.trainer import Trainer


def _dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        tokens = np.arange(128, dtype=np.uint16) % 32
        labels = np.roll(tokens, -1).astype(np.int32)
        (root / f"{split}.bin").write_bytes(tokens.tobytes())
        (root / f"{split}.labels.bin").write_bytes(labels.tobytes())
        for suffix, dtype in (("", "uint16"), (".labels", "int32")):
            (root / f"{split}{suffix}.manifest.json").write_text(
                json.dumps(
                    {
                        "dtype": dtype,
                        "files": [f"{split}{suffix}.bin"],
                        "sequence_length": 8,
                    }
                ),
                encoding="utf-8",
            )
    return root


def _config(tmp_path: Path, name: str = "run", steps: int = 4, **fields):
    packed = tmp_path / "packed"
    if not packed.exists():
        _dataset(packed)
    cfg = NanoforgeConfig(
        model=ModelConfig(vocab_size=32, max_seq_len=8, d_model=16, n_layers=1, n_heads=2),
        training=TrainConfig(
            output_dir=str(tmp_path / name),
            run_name=name,
            max_steps=steps,
            micro_batch_size=1,
            grad_accum_steps=1,
            log_interval=1,
            eval_interval=100,
            save_interval=100,
            sample_interval=0,
            warmup_steps=0,
            warmup_ratio=0,
            learning_rate=1e-3,
            min_learning_rate=1e-3,
            scheduler="constant",
        ),
        data=DataConfig(
            train_path=str(packed / "train.bin"), val_path=str(packed / "val.bin"), seq_len=8
        ),
    )
    for key, value in fields.items():
        setattr(cfg.training, key, value)
    return cfg


def _close(trainer):
    for dataset in (trainer.train_data, trainer.val_data):
        for array in getattr(dataset, "tokens", []) + (getattr(dataset, "labels", None) or []):
            array._mmap.close()


@pytest.fixture(autouse=True)
def _threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_exact_resume_matches_uninterrupted_weights_ema_and_rng(tmp_path):
    full = Trainer(_config(tmp_path, "full", 4, ema_decay=0.9))
    full.train()
    expected = {name: tensor.clone() for name, tensor in full.unwrapped_model.state_dict().items()}
    # A periodic snapshot uses the same complete schedule as uninterrupted training.
    partial = Trainer(_config(tmp_path, "partial", 4, ema_decay=0.9))
    original = partial._batch
    fetched = 0

    def stop_after_two(split, size):
        nonlocal fetched
        batch = original(split, size)
        if split == "train":
            fetched += 1
            if fetched == 2:
                partial.request_stop()
        return batch

    partial._batch = stop_after_two
    partial.train()
    saved = load_checkpoint(tmp_path / "partial" / "interrupt.pt")
    assert saved["next_step"] == 2
    resumed = Trainer(
        _config(
            tmp_path,
            "resumed",
            4,
            ema_decay=0.9,
            resume_from_checkpoint=str(tmp_path / "partial" / "interrupt.pt"),
        )
    )
    assert resumed.start_step == 2
    resumed.train()
    for name, tensor in resumed.unwrapped_model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    for name, tensor in resumed.ema.state_dict().items():
        torch.testing.assert_close(tensor, full.ema.state_dict()[name], rtol=0, atol=0)
    for item in (full, partial, resumed):
        _close(item)


def test_early_stop_records_actual_updates_and_preserves_best(tmp_path):
    trainer = Trainer(_config(tmp_path, "early", 9, eval_interval=1, early_stopping_patience=1))
    trainer.evaluate = lambda: (2.0, 7.389)
    trainer.train()
    checkpoint = load_checkpoint(tmp_path / "early" / "last.pt")
    manifest = json.loads((tmp_path / "early" / "run.manifest.json").read_text())
    assert checkpoint["next_step"] < 9
    assert manifest["resources"]["final_step"] == checkpoint["next_step"]
    assert manifest["resources"]["termination_reason"] == "early_stopped"
    resumed = Trainer(
        _config(
            tmp_path,
            "continue",
            9,
            resume_from_checkpoint=str(tmp_path / "early" / "last.pt"),
        )
    )
    assert resumed.best_val == 2.0
    assert resumed.stale_evals == 1
    _close(trainer)
    _close(resumed)


def test_failure_restores_handlers_closes_saver_and_marks_manifest(tmp_path):
    trainer = Trainer(_config(tmp_path, "failed", async_checkpoint=True))
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def fail(*args):
        raise RuntimeError("synthetic forward failure")

    trainer._forward_loss = fail
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        trainer.train()
    assert all(signal.getsignal(sig) == handler for sig, handler in before.items())
    assert trainer.checkpoints.closed
    manifest = json.loads((tmp_path / "failed" / "run.manifest.json").read_text())
    assert manifest["status"] == "failed"
    _close(trainer)


def test_checkpoint_migration_and_unique_parameter_count(tmp_path):
    cfg = _config(tmp_path)
    trainer = Trainer(cfg)
    path = tmp_path / "plain.pt"
    save_checkpoint(path, trainer.unwrapped_model, trainer.optimizer, cfg, 3)
    payload = load_checkpoint(path)
    assert payload["schema_version"] >= 3
    assert payload["next_step"] == 3
    manifest = json.loads(path.with_suffix(".pt.manifest.json").read_text())
    assert manifest["metadata"]["parameter_count"] == sum(
        p.numel() for p in trainer.model.parameters()
    )
    old = migrate_checkpoint({"schema_version": 2, "step": 3, "config": cfg, "model": {}})
    assert old["next_step"] == 4
    _close(trainer)


def test_weight_only_initialization_restarts_optimizer_budget(tmp_path):
    source = Trainer(_config(tmp_path, "source", 2))
    source.train()
    target = Trainer(
        _config(
            tmp_path,
            "target",
            3,
            init_from_checkpoint=str(tmp_path / "source" / "last.pt"),
            ema_decay=0.9,
        )
    )
    assert target.start_step == 0
    assert not target.optimizer.state
    for a, b in zip(target.unwrapped_model.parameters(), source.unwrapped_model.parameters()):
        torch.testing.assert_close(a, b)
    _close(source)
    _close(target)


def test_masked_accumulation_matches_one_token_batch(tmp_path):
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])
    y = x.clone()
    y[0, 1:] = -100
    single = Trainer(_config(tmp_path, "single", 1, micro_batch_size=2, grad_clip=0.0))
    single._batch = lambda *args: (x, y)
    single.train()
    accumulated = Trainer(_config(tmp_path, "accumulated", 1, grad_accum_steps=2, grad_clip=0.0))
    batches = iter([(x[:1], y[:1]), (x[1:], y[1:])])
    accumulated._batch = lambda *args: next(batches)
    accumulated.train()
    for a, b in zip(single.unwrapped_model.parameters(), accumulated.unwrapped_model.parameters()):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
    _close(single)
    _close(accumulated)


def test_adafactor_normalizes_factored_second_moment():
    parameter = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = Adafactor([parameter], lr=1.0, beta2=0.0)
    parameter.grad = torch.full_like(parameter, 10.0)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.full_like(parameter, -1.0))


def _distributed_worker(rank: int, store_uri: str, tmp_dir: str, backend: str):
    import torch.distributed as dist

    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=2)
    try:
        root = Path(tmp_dir)
        trainer = Trainer(_config(root, "distributed", 2, distributed_backend=backend))
        # The trainer's independent deterministic sampler does not perturb model RNG.
        batch = trainer._batch("train", 1)[0]
        torch.save(batch, root / f"rank-{rank}-batch.pt")
        trainer.train()
        torch.save(trainer.unwrapped_model.state_dict(), root / f"rank-{rank}-weights.pt")
        _close(trainer)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["ddp", "zero"])
def test_real_two_process_gloo_rank_data_checkpoint_and_gradient_sync(tmp_path, backend):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch Gloo unavailable")
    _dataset(tmp_path / "packed")
    store = (tmp_path / "gloo-store").as_uri()
    mp.spawn(_distributed_worker, args=(store, str(tmp_path), backend), nprocs=2, join=True)
    batches = [torch.load(tmp_path / f"rank-{r}-batch.pt", weights_only=True) for r in (0, 1)]
    assert not torch.equal(*batches)
    weights = [torch.load(tmp_path / f"rank-{r}-weights.pt", weights_only=True) for r in (0, 1)]
    for name in weights[0]:
        torch.testing.assert_close(weights[0][name], weights[1][name], rtol=0, atol=0)
    checkpoint = load_checkpoint(tmp_path / "distributed" / "last.pt")
    assert checkpoint["next_step"] == 2
    assert set(checkpoint["training_state"]["rank_states"]) == {0, 1}
    rows = [
        json.loads(line)
        for line in (tmp_path / "distributed" / "metrics.jsonl").read_text().splitlines()
    ]
    assert len([row for row in rows if row["event"] == "start"]) == 1


def test_recovery_budget_and_processed_token_accounting(tmp_path):
    trainer = Trainer(_config(tmp_path, "recover", 2, recovery_lr_factor=0.5))
    forward = trainer._forward_loss
    calls = 0

    def nonfinite_once(x, y):
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor(float("nan")), torch.zeros(x.shape[0], x.shape[1], 32)
        return forward(x, y)

    trainer._forward_loss = nonfinite_once
    trainer.train()
    saved = load_checkpoint(tmp_path / "recover" / "last.pt")
    assert trainer.completed_updates == 2
    assert saved["training_state"]["attempted_updates"] == 3
    assert saved["training_state"]["skipped_updates"] == 1
    assert saved["training_state"]["tokens_seen"] == 24
    assert saved["training_state"]["lr_scale"] == 0.5
    _close(trainer)
    invalid = Trainer(_config(tmp_path, "empty", 1, max_consecutive_skipped_steps=2))
    invalid._batch = lambda *args: (torch.ones(1, 8, dtype=torch.long), torch.full((1, 8), -100))
    with pytest.raises(RuntimeError, match="recovery budget"):
        invalid.train()
    assert invalid.attempted_updates == 2
    assert invalid.completed_updates == 0
    _close(invalid)


def test_swa_curriculum_offload_and_evaluation_restore_live_weights(tmp_path):
    cfg = _config(
        tmp_path,
        "averaged",
        3,
        swa_start=1,
        swa_interval=1,
        swa_eval=True,
        ema_decay=0.9,
        optimizer_cpu_offload=True,
        curriculum=[{"step": 0, "seq_len": 4}, {"step": 1, "seq_len": 8}],
    )
    trainer = Trainer(cfg)
    assert trainer._batch("train", 1)[0].shape == (1, 4)
    trainer.completed_updates = 1
    assert trainer._batch("train", 1)[0].shape == (1, 8)
    trainer.completed_updates = 0
    trainer.train()
    assert trainer.swa.updates == 3
    before = {name: value.clone() for name, value in trainer.unwrapped_model.state_dict().items()}
    loss1, _ = trainer.evaluate()
    loss2, _ = trainer.evaluate()
    assert loss1 == loss2
    for name, value in trainer.unwrapped_model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(
        value.device.type == "cpu"
        for state in trainer.optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    saved = load_checkpoint(tmp_path / "averaged" / "last.pt")
    assert saved["training_state"]["swa_updates"] == 3
    _close(trainer)


def test_health_measures_real_activations_and_concurrent_cpu_rss():
    from concurrent.futures import ThreadPoolExecutor

    from nanoforge.training.health import ActivationDiagnostics, memory_stats

    model = torch.nn.Linear(4, 4)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    diagnostics = ActivationDiagnostics(model)
    diagnostics.begin(True)
    model(torch.ones(2, 4))
    summary = diagnostics.summary()
    assert summary["health/dead_channel_fraction"] == 1
    assert summary["health/activation_std"] == 0
    diagnostics.close()
    assert not model._forward_hooks
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: memory_stats(torch.device("cpu")), range(16)))
    assert all(result["memory/process_peak_rss_mb"] > 0 for result in results)


def test_async_checkpoint_captures_immutable_training_state(tmp_path):
    from nanoforge.training.checkpoint import AsyncCheckpointSaver

    trainer = Trainer(_config(tmp_path))
    state = {"reference": torch.tensor([1.0]), "nested": {"value": 7}}
    saver = AsyncCheckpointSaver(True)
    saver.save(
        tmp_path / "async.pt",
        trainer.unwrapped_model,
        trainer.optimizer,
        trainer.config,
        2,
        training_state=state,
    )
    state["reference"].fill_(99)
    state["nested"]["value"] = 99
    saver.close()
    payload = load_checkpoint(tmp_path / "async.pt")
    assert payload["training_state"]["reference"].item() == 1
    assert payload["training_state"]["nested"]["value"] == 7
    assert saver.closed
    _close(trainer)


def test_world_size_without_synchronized_backend_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(ValueError, match="explicit synchronized"):
        Trainer(_config(tmp_path))
