from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


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

    run_dirs = sorted([d for d in skill_base.iterdir() if d.is_dir()],
                      key=lambda p: p.name,
                      reverse=True)

    for run_dir in run_dirs:
        m = run_dir / "metrics.json"
        if m.exists():
            try:
                return json.loads(m.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None
