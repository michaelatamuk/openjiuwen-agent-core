# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Level 2 Thompson Sampling — Example Selector.

Picks which training examples are handed to GEPA for each evolution run.

┌──────────────────────────┬─────────────────────────────────────────────┐
│ SequentialExampleSelector│ Returns the full trainset in its original   │
│  (legacy)                │ order — preserves current behaviour.        │
├──────────────────────────┼─────────────────────────────────────────────┤
│ ThompsonExampleSelector  │ Each example has a Beta(α, β) arm keyed by  │
│  (TS)                    │ a hash of its task_input.  select() draws a │
│                          │ Beta sample per arm and returns the top-k   │
│                          │ examples by sample value.  update() adjusts │
│                          │ the arms based on per-example GEPA fitness. │
└──────────────────────────┴─────────────────────────────────────────────┘

Factory
-------
    make_example_selector(trainset, skill_name, config) → ExampleSelectorProtocol

The factory reads ``config.ts_example_selector`` to pick the implementation.
When TS is active it also reads ``config.ts_example_batch_size`` (0 = all)
and ``config.ts_state_dir`` for the arm state persistence directory.

Arm state persists per-skill to
``<ts_state_dir>/ts_examples_<skill_name>.json``.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig


# ── Per-example Beta arm ──────────────────────────────────────────────────────

def _example_key(example) -> str:
    """Short deterministic key derived from an example's task_input."""
    text = getattr(example, "task_input", None) or str(example)
    # Use a stable 10-digit unsigned hash so keys remain short in JSON
    return str(abs(hash(text)) % (10 ** 10))


@dataclass
class _ExampleArm:
    key: str
    alpha: float = 1.0   # prior + high-fitness hits
    beta: float = 1.0    # prior + low-fitness misses

    def update(self, fitness: float, threshold: float = 0.5) -> None:
        if fitness >= threshold:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


# ── Sequential (legacy) ───────────────────────────────────────────────────────

class SequentialExampleSelector:
    """Return all training examples unchanged — preserves current behaviour.

    ``update()`` is a no-op; no state is persisted.
    """

    def __init__(self, trainset: List) -> None:
        self._trainset = list(trainset)

    def select(self) -> List:
        return list(self._trainset)

    def update(self, examples: List, fitnesses: List[float]) -> None:
        pass  # no-op


# ── Thompson Sampling example selector ────────────────────────────────────────

class ThompsonExampleSelector:
    """Select training examples via Thompson Sampling.

    On each ``select()`` call a Beta sample is drawn for every example and
    the top-*batch_size* examples are returned.  After GEPA completes,
    ``update()`` adjusts the Beta arms based on the per-example fitness
    scores produced by the inner ``skill_fitness_metric``.

    Over repeated runs on the same skill the selector learns which examples
    are most discriminating and concentrates the GEPA budget on them.

    Arm state persists to ``<state_dir>/ts_examples_<skill_name>.json``.
    """

    _STATE_PREFIX = "ts_examples_"

    def __init__(
        self,
        trainset: List,
        skill_name: str,
        batch_size: int,
        state_dir: Path,
    ) -> None:
        self._trainset = list(trainset)
        self._skill_name = skill_name
        # 0 or larger-than-set means use the full set
        self._batch_size = (
            min(batch_size, len(trainset)) if 0 < batch_size < len(trainset)
            else len(trainset)
        )
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._arms: Dict[str, _ExampleArm] = {}
        self._load()
        # Ensure every current example has an arm (new examples get Beta(1,1))
        for ex in self._trainset:
            k = _example_key(ex)
            self._arms.setdefault(k, _ExampleArm(key=k))

    # ── Persistence ──────────────────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self._state_dir / f"{self._STATE_PREFIX}{self._skill_name}.json"

    def _load(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for key, d in data.items():
                self._arms[key] = _ExampleArm(
                    key=key,
                    alpha=float(d.get("alpha", 1.0)),
                    beta=float(d.get("beta", 1.0)),
                )
        except Exception:
            pass

    def _save(self) -> None:
        path = self._state_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {key: {"alpha": arm.alpha, "beta": arm.beta}
                 for key, arm in self._arms.items()},
                indent=2,
            )
        )
        tmp.rename(path)

    # ── Public interface ──────────────────────────────────────────────────────

    def select(self) -> List:
        """Return the top-*batch_size* examples ranked by Beta sample."""
        ranked = sorted(
            self._trainset,
            key=lambda ex: self._arms.get(_example_key(ex), _ExampleArm(key="")).sample(),
            reverse=True,
        )
        return ranked[: self._batch_size]

    def update(self, examples: List, fitnesses: List[float]) -> None:
        """Adjust Beta arms based on per-example fitness scores.

        *examples* and *fitnesses* correspond to the list returned by the
        most recent ``select()`` call (same order, same length).
        """
        for ex, fitness in zip(examples, fitnesses):
            key = _example_key(ex)
            arm = self._arms.setdefault(key, _ExampleArm(key=key))
            arm.update(fitness)
        self._save()


# ── Factory ───────────────────────────────────────────────────────────────────

def make_example_selector(
    trainset: List,
    skill_name: str,
    config: "EvolverConfig",
) -> "SequentialExampleSelector | ThompsonExampleSelector":
    """Return the correct example selector based on ``config.ts_example_selector``.

    Parameters
    ----------
    trainset:
        Full list of DSPy training examples for this skill.
    skill_name:
        Used as part of the arm-state filename so history is per-skill.
    config:
        EvolverConfig instance.  Read fields: ``ts_example_selector``,
        ``ts_example_batch_size``, ``ts_state_dir``, ``output_dir``.

    Usage (inside stage05_gepa_optimizer)::

        selector = make_example_selector(trainset, skill_name, config)
        selected  = selector.select()

        # …run GEPA on `selected`…

        # After GEPA, evaluate evolved module on the selected examples:
        fitnesses = [
            skill_fitness_metric(ex, evolved_module(task_input=ex.task_input))
            for ex in selected
        ]
        selector.update(selected, fitnesses)
    """
    if getattr(config, "ts_example_selector", False):
        raw_state_dir = getattr(config, "ts_state_dir", None)
        state_dir: Path = (
            Path(raw_state_dir) if raw_state_dir else Path(config.output_dir)
        )
        batch_size = int(getattr(config, "ts_example_batch_size", 0))
        return ThompsonExampleSelector(
            trainset=trainset,
            skill_name=skill_name,
            batch_size=batch_size,
            state_dir=state_dir,
        )
    return SequentialExampleSelector(trainset)
