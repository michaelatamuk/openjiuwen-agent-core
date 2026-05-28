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

from typing import List, Optional

from evolvers.skill_evolver_single import evolve_single_skill
from openjiuwen.agent_evolving_hermess.offline.config import EvolverConfig


def evolve_skills_batch(
    skill_names: List[str],
    eval_source: str = "synthetic",
    external_sources: Optional[list] = None,
    iterations: Optional[int] = None,
    config: Optional[EvolverConfig] = None,
    reuse_dataset: bool = False,
    min_improvement: float = 0.0,
) -> List[dict]:
    """Evolve multiple skills sequentially.

    Returns a list of metrics dicts (one per skill).  If a skill fails,
    its entry contains ``{"skill_name": name, "error": "<message>"}``.

    Args:
        skill_names: List of skill names to evolve.
        (all other args: same as evolve())

    Returns:
        List of metrics dicts in the same order as skill_names.
    """
    results = []
    for name in skill_names:
        try:
            m = evolve_single_skill(
                skill_name=name,
                eval_source=eval_source,
                external_sources=external_sources,
                iterations=iterations,
                config=config,
                reuse_dataset=reuse_dataset,
                min_improvement=min_improvement,
            )
        except Exception as exc:
            m = {"skill_name": name, "error": str(exc)}
        results.append(m)
    return results
