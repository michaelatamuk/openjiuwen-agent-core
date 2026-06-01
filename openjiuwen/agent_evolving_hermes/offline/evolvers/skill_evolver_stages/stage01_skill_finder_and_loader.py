from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from offline.evolvers.config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.skills import find_skill, load_skill


def find_and_load_skill(
    skill_name: str,
    config: EvolverConfig,
    console,
) -> Tuple[Dict, Optional[dict]]:
    """Find and load the named skill; load prior-run metrics for cross-run context.

    Raises FileNotFoundError if the skill does not exist.
    Prints a one-line confirmation and, when available, the prior-run scores.
    Returns (skill_dict, prior_metrics) where prior_metrics is None on the first run.
    """
    skill_path = find_skill(skill_name, config.skills_root)
    if skill_path is None:
        raise FileNotFoundError(
            f"Skill '{skill_name}' not found under {config.skills_root}"
        )
    skill = load_skill(skill_path)
    console.print(
        f"[bold]Loaded skill[/bold] '{skill['name']}' ({len(skill['raw'])} chars)"
    )

    prior_metrics = _load_prior_metrics(skill_name, config.output_dir)
    if prior_metrics:
        console.print(
            f"[dim]Prior run: baseline={prior_metrics['baseline_score']:.4f} "
            f"evolved={prior_metrics['evolved_score']:.4f} "
            f"({prior_metrics['timestamp']})[/dim]"
        )

    return skill, prior_metrics


def _load_prior_metrics(skill_name: str, output_dir: Path) -> Optional[dict]:
    """Return the most recent metrics.json for this skill from a prior run.

    Scans timestamped run directories under ``output_dir / skill_name``,
    returning the metrics dict from the most recent run that has a
    ``metrics.json`` file.  Returns None if no prior runs exist.

    Used to detect cross-run regressions: compare the current run's
    ``baseline_score`` against the prior run's ``evolved_score`` to see
    whether a previously-evolved skill has deteriorated.
    """
    skill_base = output_dir / skill_name
    if not skill_base.exists():
        return None
    run_dirs = sorted(
        [d for d in skill_base.iterdir() if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        m = run_dir / "metrics.json"
        if m.exists():
            try:
                return json.loads(m.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None
