# coding: utf-8
"""DemoParams — single parameters object passed to run_demo."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

# Valid run-mode identifiers understood by run_demo.
# Use a subset (or empty list) to run only selected passes.
ALL_MODES: List[str] = ["no_ts", "l2_only", "l3_only", "l2_l3"]


@dataclass
class DemoParams:
    """All parameters for a single demo run.

    Parameters
    ----------
    skill_name:
        Name of the skill (used as directory key and display label).
    skill_body:
        Body text of the SKILL.md (after the ``---`` frontmatter block).
    skill_frontmatter:
        YAML content between the ``---`` delimiters.
    golden_examples:
        List of golden example dicts.  Each must have ``task_input``,
        ``expected_behavior``, ``difficulty``, and ``source``.
    model:
        DSPy model string, e.g. ``"deepseek/deepseek-chat"``.
    workdir:
        Root working directory.  All output subdirectories are derived
        from this path (see properties below).
    iterations:
        Number of GEPA optimisation iterations per training pass.
    ts_batch_size:
        Number of examples the Thompson Sampling example-selector picks
        per batch (used by ``l2_only`` and ``l2_l3`` modes).
    verbose:
        ``True`` → show full DSPy INFO training logs; ``False`` (default)
        suppresses them.
    run_modes:
        Which GEPA training passes to include in this run.
        Each string corresponds to one pass:

        * ``"no_ts"``   — plain GEPA, all examples, threshold acceptance
        * ``"l2_only"`` — TS Example Selector only
        * ``"l3_only"`` — TS Acceptance Gate only
        * ``"l2_l3"``   — both TS levels active

        Pass ``[]`` to evaluate the baseline on holdout without any training.
        Defaults to all four passes.
    """

    # ── Scenario ─────────────────────────────────────────────────────────────
    skill_name: str
    skill_body: str
    skill_frontmatter: str
    golden_examples: List[Dict[str, Any]]

    # ── LLM ──────────────────────────────────────────────────────────────────
    model: str

    # ── Paths ─────────────────────────────────────────────────────────────────
    workdir: Path

    # ── GEPA config ───────────────────────────────────────────────────────────
    iterations: int = 5
    ts_batch_size: int = 4
    verbose: bool = False
    run_modes: List[str] = field(default_factory=lambda: list(ALL_MODES))

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        # Ensure TS state dir exists immediately (multiple passes share it)
        self.ts_state_dir.mkdir(parents=True, exist_ok=True)

    # ── Derived output directories ────────────────────────────────────────────

    @property
    def skills_root(self) -> Path:
        return self.workdir / "skills"

    @property
    def output_baseline(self) -> Path:
        """Dedicated output directory for the pre-training baseline evaluation."""
        return self.workdir / "output_baseline"

    @property
    def output_no_ts(self) -> Path:
        return self.workdir / "output_no_ts"

    @property
    def output_l2_only(self) -> Path:
        return self.workdir / "output_l2_only"

    @property
    def output_l3_only(self) -> Path:
        return self.workdir / "output_l3_only"

    @property
    def output_l2_l3(self) -> Path:
        return self.workdir / "output_l2_l3"

    @property
    def ts_state_dir(self) -> Path:
        return self.workdir / "ts_state"
