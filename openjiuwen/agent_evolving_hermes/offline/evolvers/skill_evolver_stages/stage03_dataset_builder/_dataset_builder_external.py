from __future__ import annotations

from typing import List, Optional

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.external_importers import build_dataset_from_external
from ._dataset_builder_synthetic import build as build_dataset_synthetic


def build(skill_name: str, skill_raw: str, external_sources: Optional[List[str]], dataset_dir, config: EvolverConfig, console):
    console.print("[bold]Building dataset from external session logs…[/bold]")
    sources = external_sources or ["jiuwen", "claude-code"]
    dataset = build_dataset_from_external(skill_name=skill_name,
                                          skill_text=skill_raw,
                                          sources=sources,
                                          output_path=dataset_dir,
                                          model=config.eval_model)
    if not dataset.train:
        console.print("[yellow]WARNING: No external examples — falling back to synthetic.[/yellow]")
        dataset = build_dataset_synthetic(skill_raw, dataset_dir, config, console)

    return dataset
