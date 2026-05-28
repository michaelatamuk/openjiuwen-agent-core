from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from evolvers._cached_dataset_finder import _find_cached_dataset
from openjiuwen.agent_evolving_hermess.offline.config import EvolverConfig
from openjiuwen.agent_evolving_hermess.offline.dataset_builder import (
    EvalDataset,
    GoldenDatasetLoader,
    SyntheticDatasetBuilder,
)
from openjiuwen.agent_evolving_hermess.offline.external_importers import (
    build_dataset_from_external,
)


def build_or_load_dataset(
    skill_name: str,
    skill_raw: str,
    eval_source: str,
    external_sources: Optional[List[str]],
    config: EvolverConfig,
    output_dir: Path,
    reuse_dataset: bool,
    console,
) -> Tuple[EvalDataset, Optional[Path]]:
    """Build a new eval dataset or reuse the most recent cached one.

    Handles all four eval_source branches: cached, synthetic, external, golden.
    Returns (dataset, cached_path) where cached_path is non-None iff a cached
    dataset was reused.
    Raises ValueError if the resulting training split is empty.
    """
    dataset_dir = output_dir / "dataset"
    cached_path = (
        _find_cached_dataset(skill_name, config.output_dir) if reuse_dataset else None
    )

    if cached_path is not None:
        console.print(
            f"[cyan]Reusing cached dataset from {cached_path.parent.name}[/cyan]"
        )
        dataset = EvalDataset.load(cached_path)

    elif eval_source == "synthetic":
        console.print("[bold]Building synthetic dataset…[/bold]")
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(skill_raw, artifact_type="skill")
        dataset.save(dataset_dir)

    elif eval_source == "external":
        console.print("[bold]Building dataset from external session logs…[/bold]")
        sources = external_sources or ["jiuwen", "claude-code"]
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill_raw,
            sources=sources,
            output_path=dataset_dir,
            model=config.eval_model,
        )
        if not dataset.train:
            console.print(
                "[yellow]WARNING: No external examples — falling back to synthetic.[/yellow]"
            )
            builder = SyntheticDatasetBuilder(config)
            dataset = builder.generate(skill_raw, artifact_type="skill")
            dataset.save(dataset_dir)

    elif eval_source == "golden":
        golden_path = config.skills_root / skill_name / "golden_dataset"
        if not golden_path.exists():
            raise FileNotFoundError(f"No golden dataset at {golden_path}")
        dataset = GoldenDatasetLoader.load(golden_path)

    else:
        raise ValueError(f"Unknown eval_source '{eval_source}'")

    console.print(
        f"Dataset: [bold]train={len(dataset.train)}[/bold] "
        f"val={len(dataset.val)} holdout={len(dataset.holdout)}"
    )
    if not dataset.train:
        raise ValueError("Empty training set — cannot evolve.")

    return dataset, cached_path
