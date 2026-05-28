# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Configuration dataclasses for offline (GEPA skill evolver) track."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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

    # ── Trajectory dataset (eval_source="trajectory") ────────────────────────
    trajectory_dir: Optional[Path] = field(default=None)  # Folder of saved trajectory JSON files
    trajectory_min_reward: float = 0.0             # Skip steps with reward below this value

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("./skill_evolver_output"))
    create_pr: bool = False                        # Create a git PR with result?
