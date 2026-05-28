# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Main GEPA evolution orchestration.

Mirrors hermes-agent-self-evolution evolve_skill.py.

New vs original plan:
  - Rich console output (Console, Panel, Table) — matches Hermess exactly
  - Dataset caching: if a dataset already exists for this skill, reuse it
    instead of regenerating (saves LLM cost on repeated runs)
  - Cross-run metrics history: each run appends to metrics_history.jsonl
    so regressions across runs are detectable
  - min_improvement acceptance gate: if improvement < threshold, the evolved
    skill is saved as evolved_REGRESSION.md and a warning is printed, but
    it is NOT written to evolved_skill.md (avoids regressing active skills)
"""
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
