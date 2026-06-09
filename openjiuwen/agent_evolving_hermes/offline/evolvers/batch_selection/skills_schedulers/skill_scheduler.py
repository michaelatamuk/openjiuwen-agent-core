# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Level 1 Thompson Sampling — Skill Scheduler.

Decides the priority order in which skills are evolved during a batch run.

┌─────────────────────────┬──────────────────────────────────────────────┐
│ RoundRobinSkillScheduler│ Preserves the existing behaviour: skills     │
│  (legacy)               │ are returned in registration order.          │
├─────────────────────────┼──────────────────────────────────────────────┤
│ ThompsonSkillScheduler  │ Ranks skills by a single Beta(α, β) sample   │
│  (TS)                   │ drawn at schedule time.  Skills with more    │
│                         │ improvement history are preferred while      │
│                         │ unexplored skills still get a chance.        │
└─────────────────────────┴──────────────────────────────────────────────┘

Factory
-------
    make_skill_scheduler(config, skill_names) → SkillSchedulerProtocol

The factory reads ``config.ts_skill_scheduler`` to pick the implementation.

Arm state persists to ``<ts_state_dir>/ts_skill_scheduler.json`` (default
location: ``config.skills_root``).  On first use the state is bootstrapped
from existing ``metrics.json`` files found anywhere under ``skills_root``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from .round_robin import RoundRobinSkillScheduler
from .thompson import ThompsonSkillScheduler


# ── Factory ───────────────────────────────────────────────────────────────────

def make_skill_scheduler(config: EvolverConfig, skill_names: List[str]):
    """Return the correct scheduler based on ``config.ts_skill_scheduler``.

    Also calls ``register(skill_names)`` so the returned scheduler is ready
    to use immediately.

    Parameters
    ----------
    config:
        EvolverConfig instance.  Read fields: ``ts_skill_scheduler``,
        ``skills_root``, ``ts_state_dir``.
    skill_names:
        The complete list of skills that will be evolved in this batch.

    Usage::

        scheduler = make_skill_scheduler(config, all_skill_names)
        ordered   = scheduler.schedule(all_skill_names)
        for name in ordered:
            result = evolve_single_skill(name, config=config)
            scheduler.record(name, result["improvement"])
    """
    if getattr(config, "ts_skill_scheduler", False):
        state_dir: Optional[Path] = getattr(config, "ts_state_dir", None)
        scheduler: ThompsonSkillScheduler = ThompsonSkillScheduler(skills_root=config.skills_root,
                                                                   state_dir=state_dir,)
    else:
        scheduler = RoundRobinSkillScheduler()

    scheduler.register(skill_names)
    return scheduler
