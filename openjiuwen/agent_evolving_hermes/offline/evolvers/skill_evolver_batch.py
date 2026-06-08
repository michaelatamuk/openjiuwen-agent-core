# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Main GEPA evolution orchestration.

Mirrors hermes-agent-self-evolution evolve_skill.py.

New vs original plan:
  - Rich console output (Console, Panel, Table) — matches Hermes exactly
  - Dataset caching: if a dataset already exists for this skill, reuse it
    instead of regenerating (saves LLM cost on repeated runs)
  - Cross-run metrics history: each run appends to metrics_history.jsonl
    so regressions across runs are detectable
  - min_improvement acceptance gate: if improvement < threshold, the evolved
    skill is saved as evolved_REGRESSION.md and a warning is printed, but
    it is NOT written to evolved_skill.md (avoids regressing active skills)
  - Thompson Sampling skill scheduler (Level 1): when config.ts_skill_scheduler
    is True, skills are evolved in priority order determined by a Beta(α, β)
    sample drawn per skill.  Skills with a stronger improvement track record
    are scheduled first; unexplored skills still surface occasionally.
"""
from __future__ import annotations

from typing import List, Optional

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from .skill_evolver_single import evolve_single_skill
from .skill_evolver_prereqs import build_evolution_prereqs
from .selection import make_skill_scheduler
from .skill_evolver_single_params import SkillEvolverParams


def evolve_skills_batch(
    skill_names: List[str],
    eval_source: str = "synthetic",
    external_sources: Optional[list] = None,
    config: EvolverConfig = None,
    reuse_dataset: bool = False,
    min_improvement: float = 0.0,
) -> List[dict]:
    """Evolve multiple skills, optionally using TS-based priority ordering.

    Returns a list of metrics dicts (one per skill).  If a skill fails,
    its entry contains ``{"skill_name": name, "error": "<message>"}``.

    When ``config.ts_skill_scheduler`` is True the skills are run in the
    order determined by a single Thompson Sampling draw (highest expected
    improvement first).  Outcomes are recorded after each run so that state
    accumulates across multiple ``--all`` invocations.

    Args:
        skill_names: List of skill names to evolve.
        (all other args: same as evolve())

    Returns:
        List of metrics dicts in the order the skills were actually run.
    """
    # ── Level 1: schedule skill order via factory ─────────────────────────────
    scheduler = make_skill_scheduler(config, list(skill_names))
    ordered_names = scheduler.schedule(list(skill_names))

    results = []
    for name in ordered_names:
        try:
            prereqs = build_evolution_prereqs(
                skill_name=name,
                config=config,
                eval_source=eval_source,
                external_sources=external_sources,
                reuse_dataset=reuse_dataset,
            )
            params: SkillEvolverParams = SkillEvolverParams(
                skill_name=name,
                eval_source=eval_source,
                external_sources=external_sources,
                reuse_dataset=reuse_dataset,
                min_improvement=min_improvement,
                prior_metrics=prereqs.prior_metrics,
                cached_path=prereqs.cached_path,
                config=config,
                console=prereqs.console,
                prebuilt_skill=prereqs.skill,
                prebuilt_dataset=prereqs.dataset,
                prebuilt_baseline_module=prereqs.baseline_module,
                prebuilt_trainset=prereqs.trainset,
                prebuilt_valset=prereqs.valset,
            )
            m = evolve_single_skill(params)
        except Exception as exc:
            m = {"skill_name": name, "error": str(exc)}

        # Record outcome so TS arm history persists for future batch runs
        improvement = m.get("improvement", 0.0) if "error" not in m else 0.0
        scheduler.record(name, improvement)

        results.append(m)
    return results
