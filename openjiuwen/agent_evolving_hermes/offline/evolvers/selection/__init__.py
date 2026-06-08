# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Thompson Sampling selection layer for the offline GEPA skill evolver.

Three independent decision points, each with a legacy (deterministic)
implementation and a Thompson Sampling implementation chosen via config:

Level 1 — Skill Scheduler  (config.ts_skill_scheduler)
    Which skill to evolve next in a batch run.
    make_skill_scheduler(config, skill_names) → SkillSchedulerProtocol

Level 2 — Example Selector  (config.ts_example_selector)
    Which training examples to pass to GEPA.
    make_example_selector(trainset, skill_name, config) → ExampleSelectorProtocol

Level 3 — Acceptance Gate  (config.ts_acceptance_gate)
    Whether to deploy an evolved skill candidate.
    make_acceptance_gate(config, min_improvement) → AcceptanceGateProtocol
"""
from openjiuwen.agent_evolving_hermes.offline.evolvers.selection.skills_schedulers.protocols import (
    AcceptanceGateProtocol,
    ExampleSelectorProtocol,
    SkillSchedulerProtocol,
)
from openjiuwen.agent_evolving_hermes.offline.evolvers.selection.skills_schedulers.skill_scheduler import (
    RoundRobinSkillScheduler,
    ThompsonSkillScheduler,
    make_skill_scheduler,
)
from openjiuwen.agent_evolving_hermes.offline.evolvers.selection.example_selectors.selector import (
    SequentialExampleSelector,
    ThompsonExampleSelector,
    make_example_selector,
)
from openjiuwen.agent_evolving_hermes.offline.evolvers.selection.acceptance_gates.acceptance_gate import (
    ThresholdAcceptanceGate,
    ThompsonAcceptanceGate,
    make_acceptance_gate,
)

__all__ = [
    # Protocols
    "SkillSchedulerProtocol",
    "ExampleSelectorProtocol",
    "AcceptanceGateProtocol",
    # Level 1
    "RoundRobinSkillScheduler",
    "ThompsonSkillScheduler",
    "make_skill_scheduler",
    # Level 2
    "SequentialExampleSelector",
    "ThompsonExampleSelector",
    "make_example_selector",
    # Level 3
    "ThresholdAcceptanceGate",
    "ThompsonAcceptanceGate",
    "make_acceptance_gate",
]
