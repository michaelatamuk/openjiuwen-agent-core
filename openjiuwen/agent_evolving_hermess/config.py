# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Configuration dataclasses for both the online (BackgroundReviewRail)
and offline (GEPA skill evolver) tracks."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class BackgroundReviewConfig:
    """All tunable parameters for the background review rail.

    Mirrors Hermess cli-config.yaml [memory] and [skills] sections.
    """
    enabled: bool = True

    # ── Trigger thresholds ──────────────────────────────────────────────────
    skill_nudge_interval: int = 10
    """Trigger skill review after this many tool-call completions (0 = disabled)."""

    memory_nudge_interval: int = 10
    """Trigger memory review after this many user turns (0 = disabled)."""

    # ── Memory limits ────────────────────────────────────────────────────────
    memory_char_limit: int = 2200
    """Maximum character count for agent memory store (~800 tokens)."""

    user_char_limit: int = 1375
    """Maximum character count for user profile store (~500 tokens)."""

    # ── Skill constraints ────────────────────────────────────────────────────
    max_skill_size: int = 100_000
    """Maximum characters allowed in a single SKILL.md file."""

    max_skill_growth_ratio: float = 0.20
    """Background review edits may not grow a skill by more than 20%."""

    # ── Execution budget ─────────────────────────────────────────────────────
    review_max_iterations: int = 16
    """Maximum LLM iterations inside one background review run."""

    review_timeout_seconds: float = 120.0
    """Hard wall-clock timeout for a background review task."""

    # ── Skill storage ─────────────────────────────────────────────────────────
    skills_root: Optional[Path] = None
    """Root directory for SKILL.md files. Defaults to ~/.jiuwen/skills/"""

    memory_root: Optional[Path] = None
    """Root directory for memory files. Defaults to ~/.jiuwen/memories/"""

    # ── Model ────────────────────────────────────────────────────────────────
    review_model: Optional[str] = None
    """LLM model for background review. None = inherit from parent agent."""

    # ── Guards ───────────────────────────────────────────────────────────────
    protected_skill_names: List[str] = field(default_factory=list)
    """Skill names that background review must never modify (immutable skills)."""

    flush_min_turns: int = 6
    """Only trigger a review on exit if session had at least this many user turns."""


@dataclass
class EvolverConfig:
    """Configuration for one GEPA skill evolution run.

    Mirrors hermes-agent-self-evolution EvolutionConfig exactly.
    Only difference: paths default to ~/.jiuwen/ instead of ~/.hermes/.
    """
    # ── Skill storage ─────────────────────────────────────────────────────────
    skills_root: Path = field(default_factory=lambda: Path.home() / ".jiuwen" / "skills")

    # ── GEPA optimisation ─────────────────────────────────────────────────────
    iterations: int = 10
    population_size: int = 5

    # ── LLM models ────────────────────────────────────────────────────────────
    optimizer_model: str = "openai/gpt-4.1"       # Used by GEPA for reflections
    eval_model: str = "openai/gpt-4.1-mini"        # Used for LLM-as-judge scoring
    judge_model: str = "openai/gpt-4.1"            # Used for dataset generation

    # ── Constraints ───────────────────────────────────────────────────────────
    max_skill_size: int = 15_000                   # 15 KB
    max_prompt_growth: float = 0.20                # 20% max growth over baseline

    # ── Eval dataset ─────────────────────────────────────────────────────────
    eval_dataset_size: int = 20
    train_ratio: float = 0.50
    val_ratio: float = 0.25
    holdout_ratio: float = 0.25

    # ── Benchmark gating ─────────────────────────────────────────────────────
    run_pytest: bool = False                       # Run pytest after evolution?
    pytest_timeout: int = 300                      # pytest timeout in seconds

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("./skill_evolver_output"))
    create_pr: bool = False                        # Create a git PR with result?
