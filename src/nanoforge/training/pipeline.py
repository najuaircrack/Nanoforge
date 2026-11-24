from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanoforge.config import NanoforgeConfig, load_config, save_config
from nanoforge.logging_utils import get_logger

logger = get_logger("training.pipeline")


@dataclass
class StageSpec:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    kind: str = "train"
    preference_data: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.name in {".", ".."}
            or any(ch in self.name for ch in "/\\")
            or Path(self.name).is_absolute()
        ):
            raise ValueError("stage name must be a single relative directory name")
        if self.kind not in {"train", "dpo", "distill"}:
            raise ValueError(f"unsupported pipeline trainer kind: {self.kind}")


@dataclass
class PipelineRecipe:
    base_config: str
    stages: list[StageSpec]
    output_dir: str = "runs/pipeline"


def _apply_overrides(cfg: NanoforgeConfig, overrides: dict[str, Any]) -> NanoforgeConfig:
    """Shallow-merge a nested overrides dict onto a NanoforgeConfig."""
    import copy
    import dataclasses

    cfg = copy.deepcopy(cfg)
    section_map = {
        "model": cfg.model,
        "training": cfg.training,
        "data": cfg.data,
        "inference": cfg.inference,
    }
    for section, values in overrides.items():
        obj = section_map.get(section)
        if obj is None:
            logger.warning("pipeline override: unknown section '%s', skipping", section)
            continue
        if not isinstance(values, dict):
            logger.warning(
                "pipeline override: section '%s' value must be a dict, skipping", section
            )
            continue
        valid = {f.name for f in dataclasses.fields(obj)}
        for key, val in values.items():
            if key in valid:
                setattr(obj, key, val)
            else:
                logger.warning("pipeline override: unknown field '%s.%s', skipping", section, key)
    return cfg


def run_pipeline(recipe: PipelineRecipe) -> None:
    """Execute all stages in order, handing the best checkpoint from each stage to the next."""
    from nanoforge.training.trainer import Trainer

    base_cfg = load_config(recipe.base_config)
    pipeline_dir = Path(recipe.output_dir)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    prev_checkpoint: str | None = None

    for stage in recipe.stages:
        logger.info("=== pipeline stage: %s ===", stage.name)
        cfg = _apply_overrides(base_cfg, stage.overrides)

        # Each stage gets its own output dir under the pipeline dir.
        stage_dir = pipeline_dir / stage.name
        cfg.training.run_name = stage.name
        cfg.training.output_dir = str(stage_dir)

        # Resume model weights (not optimizer) from the previous stage's best checkpoint.
        if prev_checkpoint is not None:
            cfg.training.init_from_checkpoint = prev_checkpoint
            cfg.training.resume_from_checkpoint = None
            logger.info("stage '%s': loading weights from %s", stage.name, prev_checkpoint)

        # Save the resolved config for reproducibility.
        stage_cfg_path = stage_dir / "config.yaml"
        stage_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, stage_cfg_path)

        if stage.kind == "dpo":
            from nanoforge.training.dpo_trainer import DPOTrainer

            if not stage.preference_data:
                raise ValueError(f"DPO stage {stage.name} requires preference_data")
            trainer = DPOTrainer(cfg, stage.preference_data)
        elif stage.kind == "distill":
            from nanoforge.training.distillation_trainer import DistillationTrainer

            trainer = DistillationTrainer(cfg)
        else:
            trainer = Trainer(cfg)
        trainer.train()

        best = stage_dir / "best.pt"
        last = stage_dir / "last.pt"
        prev_checkpoint = str(best) if best.exists() else str(last) if last.exists() else None
        if prev_checkpoint:
            logger.info("stage '%s' done; next stage will load: %s", stage.name, prev_checkpoint)
        else:
            raise RuntimeError(
                f"stage {stage.name!r} produced no checkpoint; refusing a scratch handoff"
            )

    logger.info("pipeline complete. stages: %s", [s.name for s in recipe.stages])


def load_recipe(path: str | Path) -> PipelineRecipe:
    """Load a pipeline recipe YAML."""
    import yaml

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    stages = [
        StageSpec(
            name=s["name"],
            overrides=s.get("overrides", {}),
            kind=s.get("kind", "train"),
            preference_data=s.get("preference_data"),
        )
        for s in raw.get("stages", [])
    ]
    return PipelineRecipe(
        base_config=raw["base_config"],
        stages=stages,
        output_dir=raw.get("output_dir", "runs/pipeline"),
    )
