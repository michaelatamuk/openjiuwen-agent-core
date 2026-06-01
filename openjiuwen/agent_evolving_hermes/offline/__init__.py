# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""openjiuwen.agent_evolving_hermes — Hermes-style self-evolution for Jiuwen.

Two independent tracks, one package:

Track — Offline (GEPA skill evolver)
  Mirrors Hermes's GEPA (Genetic Evolution of Prompt Artifacts).
  CLI tool that uses DSPy + GEPA to optimise a SKILL.md body against
  a scored evaluation dataset. Triggered manually or by cron.
"""
from __future__ import annotations

# ── Offline track ─────────────────────────────────────────────────────────────
from offline.evolvers.skill_evolver_config import EvolverConfig

from openjiuwen.agent_evolving_hermes.offline.constraints import (
    ConstraintResult,
    ConstraintValidator,
)
from openjiuwen.agent_evolving_hermes.offline.dataset_builder import (
    EvalDataset,
    EvalExample,
    GoldenDatasetLoader,
    SyntheticDatasetBuilder,
)
from .evolvers import evolve_skills_batch, evolve_single_skill
from .external_importers import (
    ClaudeCodeImporter,
    JiuwenSessionImporter,
    SECRET_PATTERNS,
    build_dataset_from_external,
)
from .evolvers.skill_evolver_stages.stage05_gepa_optimizer import (skill_fitness_metric)
from .evolvers.skill_evolver_stages.stage08_holdout_evaluator_judge import LLMJudge, FitnessScore
from openjiuwen.agent_evolving_hermes.offline.skills import (
    SkillModule,
    find_skill,
    load_skill,
    reassemble_skill,
)

__all__ = [
    # Offline track
    "EvolverConfig",
    "evolve_single_skill",
    "evolve_skills_batch",
    "SkillModule",
    "find_skill",
    "load_skill",
    "reassemble_skill",
    "FitnessScore",
    "LLMJudge",
    "skill_fitness_metric",
    "ConstraintResult",
    "ConstraintValidator",
    "EvalExample",
    "EvalDataset",
    "GoldenDatasetLoader",
    "SyntheticDatasetBuilder",
    "JiuwenSessionImporter",
    "ClaudeCodeImporter",
    "SECRET_PATTERNS",
    "build_dataset_from_external",
]
