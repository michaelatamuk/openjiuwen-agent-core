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

from pathlib import Path
from typing import List

from openjiuwen.agent_evolving_hermes.offline.evolvers.selection.example_selectors.sequential_examples import \
    SequentialExampleSelector
from openjiuwen.agent_evolving_hermes.offline.evolvers.selection.example_selectors.thompson_examples import \
    ThompsonExampleSelector
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig


# ── Factory ───────────────────────────────────────────────────────────────────

def make_example_selector(trainset: List, skill_name: str, config: EvolverConfig):
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
        state_dir: Path = (Path(raw_state_dir) if raw_state_dir else Path(config.output_dir))
        batch_size = int(getattr(config, "ts_example_batch_size", 0))
        return ThompsonExampleSelector(trainset=trainset,
                                       skill_name=skill_name,
                                       batch_size=batch_size,
                                       state_dir=state_dir)

    return SequentialExampleSelector(trainset)
