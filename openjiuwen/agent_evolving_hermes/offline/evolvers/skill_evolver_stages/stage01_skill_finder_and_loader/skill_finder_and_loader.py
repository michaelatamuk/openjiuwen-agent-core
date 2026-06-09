from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.skills import find_skill, load_skill
from skill_evolver_stages.stage01_skill_finder_and_loader._prior_metrics_loader import _load_prior_metrics


def find_and_load_skill(skill_name: str,
                        config: EvolverConfig,
                        console) -> Tuple[Dict, Optional[dict]]:
    """Find and load the named skill; load prior-run metrics for cross-run context.

    Raises FileNotFoundError if the skill does not exist.
    Prints a one-line confirmation and, when available, the prior-run scores.
    Returns (skill_dict, prior_metrics) where prior_metrics is None on the first run.
    """
    console.print("\n[blue]~~~ Evolving Stage 01 - Skill Find and Load Started ~~~[/blue]")

    skill_path = find_skill(skill_name, config.skills_root)
    if skill_path is None:
        raise FileNotFoundError(f"Skill '{skill_name}' not found under {config.skills_root}")

    skill = load_skill(skill_path)
    console.print(f"[green]Loaded skill '{skill['name']}' ({len(skill['raw'])} chars[/green])")

    prior_metrics = _load_prior_metrics(skill_name, config.output_dir)
    if prior_metrics:
        console.print(f"[dim]Prior run: baseline={prior_metrics['baseline_score']:.4f} "
                      f"evolved={prior_metrics['evolved_score']:.4f} "
                      f"({prior_metrics['timestamp']})[/dim]")

    console.print("[blue]~~~ Evolving Stage 01 - Skill Find and Load Finished ~~~[/blue]")
    return skill, prior_metrics
