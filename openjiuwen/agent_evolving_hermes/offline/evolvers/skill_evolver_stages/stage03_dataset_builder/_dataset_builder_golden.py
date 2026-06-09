from __future__ import annotations

from openjiuwen.agent_evolving_hermes.offline.dataset_builder import GoldenDatasetLoader
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig

def build(skill_name: str, config: EvolverConfig, console):
    console.print("[bold]Building golden dataset…[/bold]")
    golden_path = config.skills_root / skill_name / "golden_dataset"
    if not golden_path.exists():
        raise FileNotFoundError(f"No golden dataset at {golden_path}")
    dataset = GoldenDatasetLoader.load(golden_path)

    return dataset

