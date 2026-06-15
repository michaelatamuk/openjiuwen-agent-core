# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Configuration dataclasses for offline (GEPA skill evolver) track."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional


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

    # ── Thompson Sampling ─────────────────────────────────────────────────────
    # Each flag independently enables the TS variant for its decision level.
    # When False (default) the existing deterministic behaviour is preserved.
    ts_skill_scheduler: bool = False          # Level 1: TS batch skill ordering
    ts_example_selector: bool = False         # Level 2: TS training-example selection
    ts_example_batch_size: int = 0            # 0 = full trainset; N = top-N TS subset
    ts_acceptance_gate: bool = False          # Level 3: TS candidate acceptance
    ts_acceptance_confidence: float = 0.75   # Min TS win-prob to accept (Level 3)
    ts_acceptance_n_samples: int = 100        # Monte Carlo draws for TS confidence
    ts_state_dir: Optional[Path] = field(default=None)  # Arm state dir (default: skills_root)

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("./skill_evolver_output"))
    create_pr: bool = False                        # Create a git PR with result?

    # ── Multi-objective scoring ───────────────────────────────────────────────
    # "holistic" — single-scalar composite score (default, unchanged behaviour)
    # "rubrics"    — 5-dimension scoring with no-regression rule + dynamic weights
    scoring_mode: str = "holistic"

    # ── Fitness metric (fast proxy used inside the GEPA optimizer loop) ───────
    # Built-ins: "f1" (stop-word-filtered F1, general-purpose, default),
    #            "bag_of_words" (word-bag with 0.3 floor),
    #            "graph" (concept-graph structural similarity).
    # Custom: dotted import path ("pkg.module.fn") or a key in custom_fitness_metrics.
    fitness_metric: str = "bag_of_words"
    custom_fitness_metrics: Dict[str, Any] = field(default_factory=dict)
    # Optional pre-resolved callable that bypasses the resolver entirely.
    # Set by the scoring-matrix mode to inject a logging-wrapped metric fn.
    fitness_metric_fn_override: Optional[Callable] = field(default=None, repr=False)

    # ── Verbosity ─────────────────────────────────────────────────────────────
    verbose: bool = False                          # True = show DSPy INFO training logs
