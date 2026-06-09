from __future__ import annotations

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.external_importers import build_dataset_from_trajectories
from ._dataset_builder_synthetic import build as build_dataset_synthetic


def build(skill_name: str, skill_raw: str, dataset_dir, config: EvolverConfig, console):
    console.print("[bold]Building dataset from saved trajectories…[/bold]")
    dataset = build_dataset_from_trajectories(skill_name=skill_name,
                                              skill_text=skill_raw,
                                              output_path=dataset_dir,
                                              model=config.eval_model,
                                              trajectory_dir=config.trajectory_dir,
                                              min_reward=config.trajectory_min_reward)
    if not dataset.train:
        console.print("[yellow]WARNING: No trajectory examples — falling back to synthetic.[/yellow]")
        dataset = build_dataset_synthetic(skill_raw, dataset_dir, config, console)

    return dataset
