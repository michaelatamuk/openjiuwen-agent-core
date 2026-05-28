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
from openjiuwen.agent_evolving_hermess.provenance import (
    background_review_context,
    get_write_origin,
    make_write_metadata,
    set_write_origin,
)
from openjiuwen.agent_evolving_hermess.review_executor import run_background_review
from openjiuwen.agent_evolving_hermess.skill_store import (
    SKILL_STATE_ACTIVE,
    SKILL_STATE_ARCHIVED,
    SKILL_STATE_STALE,
    UsageSidecar,
    build_skills_system_prompt,
    skill_archive,
    skill_create,
    skill_delete,
    skill_edit,
    skill_get_usage,
    skill_list,
    skill_patch,
    skill_read,
    skill_restore,
    skill_set_pinned,
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
from openjiuwen.agent_evolving_hermess.evolve import batch_evolve, evolve
from openjiuwen.agent_evolving_hermess.external_importers import (
    ClaudeCodeImporter,
    JiuwenSessionImporter,
    RelevanceFilter,
    SECRET_PATTERNS,
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
    # Online track — rail + config
    "BackgroundReviewRail",
    "BackgroundReviewConfig",
    "ReviewMode",
    "ReviewTrigger",
    "ReviewAction",
    "ReviewResult",
    # Memory
    "MemoryStore",
    "run_background_review",
    # Provenance — ContextVar-based write-origin tracking
    "make_write_metadata",
    "get_write_origin",
    "set_write_origin",
    "background_review_context",
    # Prompts
    "select_prompt",
    "MEMORY_REVIEW_PROMPT",
    "SKILL_REVIEW_PROMPT",
    "COMBINED_REVIEW_PROMPT",
    # Skill store — CRUD
    "skill_read",
    "skill_create",
    "skill_edit",
    "skill_patch",
    "skill_delete",
    "skill_list",
    "build_skills_system_prompt",
    # Skill store — lifecycle
    "skill_archive",
    "skill_restore",
    "skill_get_usage",
    "skill_set_pinned",
    "UsageSidecar",
    "SKILL_STATE_ACTIVE",
    "SKILL_STATE_STALE",
    "SKILL_STATE_ARCHIVED",
    # Offline track
    "EvolverConfig",
    "evolve",
    "batch_evolve",
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
    "SECRET_PATTERNS",
    "build_dataset_from_external",
]
