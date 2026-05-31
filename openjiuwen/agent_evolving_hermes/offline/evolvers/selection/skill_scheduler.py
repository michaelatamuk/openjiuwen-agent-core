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

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openjiuwen.agent_evolving_hermes.offline.config import EvolverConfig


# ── Shared Beta arm ───────────────────────────────────────────────────────────

@dataclass
class _BetaArm:
    skill_name: str
    alpha: float = 1.0   # prior + number of runs where improvement > 0
    beta: float = 1.0    # prior + number of runs where improvement ≤ 0
    n_runs: int = 0

    def update(self, improvement: float) -> None:
        self.n_runs += 1
        if improvement > 0.0:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def sample(self) -> float:
        """Draw θ ~ Beta(α, β)."""
        return random.betavariate(self.alpha, self.beta)

    @property
    def mean(self) -> float:
        """Posterior mean E[θ] = α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)


# ── Round-robin (legacy) ──────────────────────────────────────────────────────

class RoundRobinSkillScheduler:
    """Return skills in registration order — preserves current behaviour.

    ``schedule()`` returns the skill list unchanged.
    ``record()`` is a no-op (outcomes are not used).
    """

    def __init__(self) -> None:
        self._names: List[str] = []

    def register(self, skill_names: List[str]) -> None:
        for name in skill_names:
            if name not in self._names:
                self._names.append(name)

    def schedule(self, skill_names: List[str]) -> List[str]:
        """Return *skill_names* in the order they were registered."""
        # Preserve registration order; honour any subset passed in
        registered_set = set(self._names)
        ordered = [n for n in self._names if n in set(skill_names)]
        extras = [n for n in skill_names if n not in registered_set]
        return ordered + extras

    def record(self, skill_name: str, improvement: float) -> None:
        pass  # round-robin ignores outcomes

    def rankings(self) -> List[Tuple[str, float]]:
        return [(name, 0.5) for name in self._names]


# ── Thompson Sampling scheduler ───────────────────────────────────────────────

class ThompsonSkillScheduler:
    """Prioritise skills by Thompson Sampling over Beta(α, β) distributions.

    Each skill has an arm.  ``schedule()`` draws one Beta sample per arm and
    returns skills sorted by their sample (highest first).  This means skills
    that have historically improved most are likely to be scheduled first,
    while under-explored skills still occasionally bubble to the top.

    State is persisted to ``<state_dir>/ts_skill_scheduler.json`` so arm
    history survives between CLI invocations.
    """

    _STATE_FILE = "ts_skill_scheduler.json"

    def __init__(
        self,
        skills_root: Path,
        state_dir: Optional[Path] = None,
    ) -> None:
        self._skills_root = Path(skills_root)
        self._state_dir = Path(state_dir) if state_dir else self._skills_root
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._arms: Dict[str, _BetaArm] = {}
        self._load()
        self._bootstrap_from_metrics()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self._state_dir / self._STATE_FILE

    def _load(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for name, d in data.items():
                self._arms[name] = _BetaArm(
                    skill_name=name,
                    alpha=float(d.get("alpha", 1.0)),
                    beta=float(d.get("beta", 1.0)),
                    n_runs=int(d.get("n_runs", 0)),
                )
        except Exception:
            pass

    def save(self) -> None:
        """Atomically persist arm state to disk."""
        path = self._state_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    name: {
                        "alpha": arm.alpha,
                        "beta": arm.beta,
                        "n_runs": arm.n_runs,
                    }
                    for name, arm in self._arms.items()
                },
                indent=2,
            )
        )
        tmp.rename(path)

    def _bootstrap_from_metrics(self) -> None:
        """Seed arm params from existing stage11 metrics.json files.

        Expected layout: ``<skills_root>/<skill_name>/<timestamp>/metrics.json``
        Allows the scheduler to start informed even on the very first batch run
        after Thompson Sampling is introduced (avoids a cold-start penalty for
        skills that already have evolution history).
        """
        for metrics_file in self._skills_root.rglob("metrics.json"):
            try:
                # …/<skill_name>/<timestamp>/metrics.json  → parent.parent.name
                skill_name = metrics_file.parent.parent.name
            except Exception:
                continue
            if skill_name in self._arms:
                continue  # already loaded from state file
            try:
                m = json.loads(metrics_file.read_text())
                improvement = float(m.get("improvement", 0.0))
                arm = self._arms.setdefault(skill_name, _BetaArm(skill_name=skill_name))
                arm.update(improvement)
            except Exception:
                pass

    # ── Public interface ──────────────────────────────────────────────────────

    def register(self, skill_names: List[str]) -> None:
        """Ensure every skill has an arm (cold-start = Beta(1, 1))."""
        for name in skill_names:
            self._arms.setdefault(name, _BetaArm(skill_name=name))

    def schedule(self, skill_names: List[str]) -> List[str]:
        """Return *skill_names* sorted by a single Beta sample (highest first).

        Skills with a strong improvement track-record are likely to appear
        early; unexplored skills retain a chance to be prioritised.
        """
        self.register(skill_names)
        return sorted(
            skill_names,
            key=lambda n: self._arms[n].sample(),
            reverse=True,
        )

    def record(self, skill_name: str, improvement: float) -> None:
        """Update arm for *skill_name* and persist state."""
        arm = self._arms.setdefault(skill_name, _BetaArm(skill_name=skill_name))
        arm.update(improvement)
        self.save()

    def rankings(self) -> List[Tuple[str, float]]:
        """Return (skill_name, posterior_mean) sorted best-first."""
        return sorted(
            [(name, arm.mean) for name, arm in self._arms.items()],
            key=lambda x: x[1],
            reverse=True,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def make_skill_scheduler(
    config: "EvolverConfig",
    skill_names: List[str],
) -> "RoundRobinSkillScheduler | ThompsonSkillScheduler":
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
        scheduler: RoundRobinSkillScheduler | ThompsonSkillScheduler = ThompsonSkillScheduler(
            skills_root=config.skills_root,
            state_dir=state_dir,
        )
    else:
        scheduler = RoundRobinSkillScheduler()

    scheduler.register(skill_names)
    return scheduler
