"""Distributed training helpers: environment discovery, process-group setup, FSDP/DDP wrapping.

Design goals
------------
* **No-op on a single process.** Everything degrades gracefully when the job is
  launched without ``torchrun``/``torch.distributed`` — ``is_distributed()``
  returns ``False`` and the model is returned untouched. This is the path that
  runs (and is tested) on CPU / single-GPU machines.
* **Real code on the multi-GPU path.** When launched under ``torchrun`` (env
  vars ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` present) with ``backend != "none"``
  we initialise a NCCL/Gloo process group and wrap the model in FSDP or DDP.

Launch example (single node, 4 GPUs)::

    torchrun --standalone --nproc_per_node=4 -m nanoforge train \
        --config configs/small-base.yaml

with ``training.distributed_backend: fsdp`` in the YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
from torch import nn

from nanoforge.logging_utils import get_logger

_log = get_logger("training.distributed")


@dataclass(frozen=True)
class DistEnv:
    """Snapshot of the distributed environment for the current process."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def discover_env() -> DistEnv:
    """Read the standard ``torchrun`` environment variables.

    Returns a single-process :class:`DistEnv` when they are absent, so callers can
    branch on ``env.enabled`` without worrying about whether the job was launched
    distributed.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or rank < 0 or rank >= world_size or local_rank < 0:
        raise ValueError("Distributed environment requires world_size >= 1 and valid rank/local_rank.")
    return DistEnv(rank=rank, local_rank=local_rank, world_size=world_size)


def init_process_group(env: DistEnv, backend: str | None = None) -> None:
    """Initialise the default process group if running distributed and not yet init'd."""

    import torch.distributed as dist

    if not env.enabled or dist.is_initialized():
        return
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    _log.info(
        "initialising process group: backend=%s rank=%d/%d local_rank=%d",
        backend,
        env.rank,
        env.world_size,
        env.local_rank,
    )
    dist.init_process_group(
        backend=backend, rank=env.rank, world_size=env.world_size, timeout=timedelta(seconds=60)
    )
    if backend == "nccl":
        torch.cuda.set_device(env.local_rank)


def destroy_process_group() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _mixed_precision_policy(precision: str):
    """Map Nanoforge precision strings to an FSDP MixedPrecision policy (or None)."""

    from torch.distributed.fsdp import MixedPrecision

    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    elif precision == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:  # fp32 / unknown -> no autocast in FSDP comm/compute
        return None
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def wrap_model(
    model: nn.Module,
    env: DistEnv,
    backend: str,
    *,
    precision: str = "fp32",
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Wrap ``model`` for distributed training.

    Returns the model unchanged when not running distributed or when
    ``backend == "none"`` — this is the single-process path exercised on CPU.
    For ``fsdp`` the transformer blocks are wrapped individually (transformer
    auto-wrap policy) so parameters shard at block granularity; ``ddp`` uses
    :class:`DistributedDataParallel`.
    """

    backend = (backend or "none").strip().lower()
    if not env.enabled or backend in {"none", ""}:
        return model

    if backend == "fsdp":
        return _wrap_fsdp(model, env, precision=precision)
    if backend in {"ddp", "torch_ddp", "zero"}:
        return _wrap_ddp(model, env)
    raise ValueError(
        f"Unknown distributed_backend '{backend}'. Expected one of: none, fsdp, ddp, zero."
    )


def _wrap_fsdp(model: nn.Module, env: DistEnv, *, precision: str) -> nn.Module:
    """Wrap FULL_SHARD; Trainer explicitly gathers full model/optimizer checkpoints.

    CUDA roundtrip verification is required before a hardware release claim.
    """

    import functools

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from nanoforge.model.transformer import TransformerBlock

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError(
            "FSDP requires CUDA in the supported workflow; CPU scale-out uses DDP or ZeRO stage 1."
        )
    device_id = device
    _log.info("wrapping model in FSDP (full shard) on rank %d", env.rank)
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=_mixed_precision_policy(precision),
        device_id=device_id,
        use_orig_params=True,
    )


def _wrap_ddp(model: nn.Module, env: DistEnv) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel as DDP

    if next(model.parameters()).device.type == "cuda":
        device_ids = [env.local_rank]
        output_device = env.local_rank
    else:
        device_ids = None
        output_device = None
    _log.info("wrapping model in DDP on rank %d", env.rank)
    return DDP(model, device_ids=device_ids, output_device=output_device)
