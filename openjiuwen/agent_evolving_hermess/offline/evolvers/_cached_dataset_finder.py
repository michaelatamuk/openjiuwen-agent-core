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
from pathlib import Path
from typing import Optional


def _find_cached_dataset(skill_name: str, output_dir: Path) -> Optional[Path]:
    """Return the most recent cached dataset directory for this skill, or None."""
    skill_base = output_dir / skill_name
    if not skill_base.exists():
        return None
    candidates = sorted(
        (d / "dataset" for d in skill_base.iterdir() if d.is_dir()),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "train.jsonl").exists():
            return candidate
    return None
