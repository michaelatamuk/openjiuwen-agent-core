from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.dataset_builder import (
    EvalDataset,
    GoldenDatasetLoader,
)
from skill_evolver_stages.stage03_dataset_builder._cached_dataset_finder import _find_cached_dataset
from skill_evolver_stages.stage03_dataset_builder._dataset_builder_synthetic import build as build_dataset_synthetic
from skill_evolver_stages.stage03_dataset_builder._dataset_builder_external import build as build_dataset_external
from skill_evolver_stages.stage03_dataset_builder._dataset_builder_trajectory import build as build_dataset_trajectory
from skill_evolver_stages.stage03_dataset_builder._dataset_builder_golden import build as build_dataset_golden


def build_or_load_dataset(skill_name: str,
                          skill_raw: str,
                          eval_source: str,
                          external_sources: Optional[List[str]],
                          config: EvolverConfig,
                          output_dir: Path,
                          reuse_dataset: bool,
                          console) -> Tuple[EvalDataset, Optional[Path]]:
    """Build a new eval dataset or reuse the most recent cached one.

    Handles all five eval_source branches: cached, synthetic, external, trajectory, golden.
    Returns (dataset, cached_path) where cached_path is non-None iff a cached
    dataset was reused.
    Raises ValueError if the resulting training split is empty.
    """
    console.print("\n[blue]~~~ Evolving Stage 03 - Dataset Build/Load Started ~~~[/blue]")

    dataset_dir = output_dir / "dataset"

    cached_path = (_find_cached_dataset(skill_name, config.output_dir) if reuse_dataset else None)
    if cached_path is not None:
        console.print(f"[cyan]Reusing cached dataset from {cached_path.parent.name}[/cyan]")
        dataset = EvalDataset.load(cached_path)

    elif eval_source == "synthetic":
        dataset = build_dataset_synthetic(skill_raw, dataset_dir, config, console)

    elif eval_source == "external":
        dataset = build_dataset_external(skill_name, skill_raw, external_sources, dataset_dir, config, console)

    elif eval_source == "trajectory":
        dataset = build_dataset_trajectory(skill_name, skill_raw, dataset_dir, config, console)

    elif eval_source == "golden":
        dataset = build_dataset_golden(skill_name, config, console)
    else:
        raise ValueError(f"Unknown eval_source '{eval_source}'")

    console.print(f"[green]Dataset: [bold]train={len(dataset.train)} "
                  f"val={len(dataset.val)} holdout={len(dataset.holdout)}[/green]")
    if not dataset.train:
        raise ValueError("Empty training set — cannot evolve.")

    console.print("[blue]~~~ Evolving Stage 03 - Dataset Build/Load Finished ~~~[/blue]")
    return dataset, cached_path
