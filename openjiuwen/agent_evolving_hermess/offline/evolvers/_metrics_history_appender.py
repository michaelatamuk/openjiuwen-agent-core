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

import json


def _append_metrics_history(skill_name: str, output_dir: Path, metrics: dict) -> None:
    """Append metrics to the per-skill metrics_history.jsonl file.

    This file accumulates one JSON record per run so regression trends
    can be plotted or scanned programmatically.  It lives at:
        <output_dir>/<skill_name>/metrics_history.jsonl
    """
    history_path = output_dir / skill_name / "metrics_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")
