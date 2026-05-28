# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""openjiuwen.agent_evolving_hermess — Hermess-style self-evolution for Jiuwen.

Two independent tracks, one package:

Track 1 — Online (BackgroundReviewRail)
  Mirrors Hermess background review daemon thread.
  Spawns an asyncio task after every N tool-calls or M user turns,
  reads the conversation, and uses an LLM to update SKILL.md files
  and memory entries directly.

Track 2 — Offline (GEPA skill evolver)
  Mirrors Hermess's GEPA (Genetic Evolution of Prompt Artifacts).
  CLI tool that uses DSPy + GEPA to optimise a SKILL.md body against
  a scored evaluation dataset. Triggered manually or by cron.

Neither track touches the existing agent_evolving or agent_healing systems.
"""
from __future__ import annotations

# ── Online track ──────────────────────────────────────────────────────────────
from openjiuwen.agent_evolving_hermess.background_review_rail import BackgroundReviewRail
from openjiuwen.agent_evolving_hermess.background_review_prompts import (
    COMBINED_REVIEW_PROMPT,
    MEMORY_REVIEW_PROMPT,
    SKILL_REVIEW_PROMPT,
    select_prompt,
)
from openjiuwen.agent_evolving_hermess.config import BackgroundReviewConfig, EvolverConfig
from openjiuwen.agent_evolving_hermess.memory_store import MemoryStore
from openjiuwen.agent_evolving_hermess.provenance import make_write_metadata
from openjiuwen.agent_evolving_hermess.review_executor import run_background_review
from openjiuwen.agent_evolving_hermess.skill_store import (
    skill_create,
    skill_edit,
    skill_list,
    skill_patch,
    skill_read,
)
from openjiuwen.agent_evolving_hermess.types import (
    ReviewAction,
    ReviewMode,
    ReviewResult,
    ReviewTrigger,
)

# ── Offline track ─────────────────────────────────────────────────────────────
from openjiuwen.agent_evolving_hermess.constraints import (
    ConstraintResult,
    ConstraintValidator,
)
from openjiuwen.agent_evolving_hermess.dataset_builder import (
    EvalDataset,
    EvalExample,
    GoldenDatasetLoader,
    SyntheticDatasetBuilder,
)
from openjiuwen.agent_evolving_hermess.evolve import evolve
from openjiuwen.agent_evolving_hermess.external_importers import (
    ClaudeCodeImporter,
    JiuwenSessionImporter,
    RelevanceFilter,
    build_dataset_from_external,
)
from openjiuwen.agent_evolving_hermess.fitness import (
    FitnessScore,
    LLMJudge,
    skill_fitness_metric,
)
from openjiuwen.agent_evolving_hermess.skill_module import (
    SkillModule,
    find_skill,
    load_skill,
    reassemble_skill,
)

__all__ = [
    # Online track
    "BackgroundReviewRail",
    "BackgroundReviewConfig",
    "ReviewMode",
    "ReviewTrigger",
    "ReviewAction",
    "ReviewResult",
    "MemoryStore",
    "run_background_review",
    "make_write_metadata",
    "select_prompt",
    "MEMORY_REVIEW_PROMPT",
    "SKILL_REVIEW_PROMPT",
    "COMBINED_REVIEW_PROMPT",
    "skill_read",
    "skill_create",
    "skill_edit",
    "skill_patch",
    "skill_list",
    # Offline track
    "EvolverConfig",
    "evolve",
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
    "RelevanceFilter",
    "build_dataset_from_external",
]
