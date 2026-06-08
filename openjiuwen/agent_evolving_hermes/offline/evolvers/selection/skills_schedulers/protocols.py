# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Structural protocols for the three Thompson Sampling selection levels.

Each protocol declares the interface that both the legacy (deterministic)
and the Thompson Sampling implementations must satisfy.  Concrete classes
and factory functions live in the sibling modules:

  skill_scheduler  — Level 1  (which skill to evolve next in a batch)
  example_selector — Level 2  (which training examples to pass to GEPA)
  acceptance_gate  — Level 3  (whether to deploy an evolved candidate)

Factory functions are re-exported from this package's ``__init__.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class SkillSchedulerProtocol(Protocol):
    """Decides the order in which skills are evolved in a batch run.

    Level 1 Thompson Sampling target.
    Implementations: RoundRobinSkillScheduler, ThompsonSkillScheduler.
    """

    def register(self, skill_names: List[str]) -> None:
        """Declare the full set of skills available for selection."""
        ...

    def schedule(self, skill_names: List[str]) -> List[str]:
        """Return the full list of skills in the order they should be run.

        Each name appears exactly once.  The list may be a re-ordering of
        *skill_names* (Thompson) or the same order (round-robin).
        """
        ...

    def record(self, skill_name: str, improvement: float) -> None:
        """Notify the scheduler of the outcome after a run completes.

        Implementations may persist state so that future batch runs benefit
        from the accumulated history.
        """
        ...

    def rankings(self) -> List[Tuple[str, float]]:
        """Return (skill_name, expected_value) pairs sorted best-first."""
        ...


@runtime_checkable
class ExampleSelectorProtocol(Protocol):
    """Picks the training examples to use for a single GEPA call.

    Level 2 Thompson Sampling target.
    Implementations: SequentialExampleSelector, ThompsonExampleSelector.
    """

    def select(self) -> List:
        """Return the (sub-)set of training examples to pass to GEPA."""
        ...

    def update(self, examples: List, fitnesses: List[float]) -> None:
        """Feed back per-example fitness scores after GEPA completes.

        *examples* and *fitnesses* must be the same length and correspond
        to the examples returned by the most recent ``select()`` call.
        """
        ...


@runtime_checkable
class AcceptanceGateProtocol(Protocol):
    """Decides whether to deploy an evolved skill candidate.

    Level 3 Thompson Sampling target.
    Implementations: ThresholdAcceptanceGate, ThompsonAcceptanceGate.
    """

    def decide(
        self,
        improvement: float,
        evolved_score: float,
        skill_name: str,
        evolved_text: str,
        cross_run_delta: Optional[float],
        output_dir: Path,
        console,
    ) -> Tuple[bool, Optional[float]]:
        """Decide whether to accept the evolved candidate.

        Returns
        -------
        accepted : bool
            True  → deploy (write evolved_skill.md)
            False → reject (write evolved_REGRESSION.md, no deploy)
        ts_confidence : float or None
            For the Thompson gate: P(candidate > deployed) estimated via
            Monte Carlo.  Always None for the threshold gate.
        """
        ...
