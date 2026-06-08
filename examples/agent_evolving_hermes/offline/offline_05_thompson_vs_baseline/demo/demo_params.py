# coding: utf-8
"""DemoParams — per-scenario data for a single demo run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DemoParams:
    """Scenario-specific data for one run.

    Shared settings (model, iterations, run modes, …) live in
    :class:`~demo.demo_config.DemoConfig` and are loaded from
    ``config.json`` by :func:`demo.config_loader.load_config`.

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
    workdir:
        Root working directory.  All output subdirectories are derived
        from this path (see properties below).
    """

    # ── Scenario ─────────────────────────────────────────────────────────────
    skill_name: str
    skill_body: str
    skill_frontmatter: str
    golden_examples: List[Dict[str, Any]]

    # ── Paths ─────────────────────────────────────────────────────────────────
    workdir: Path

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
    def output_gepa_plain_holistic(self) -> Path:
        return self.workdir / "output_gepa_plain_holistic"

    @property
    def output_gepa_focused_on_difficulty(self) -> Path:
        return self.workdir / "output_gepa_focused_on_difficulty"

    @property
    def output_gepa_gated(self) -> Path:
        return self.workdir / "output_gepa_gated"

    @property
    def output_gepa_full(self) -> Path:
        return self.workdir / "output_gepa_full"

    @property
    def ts_state_dir(self) -> Path:
        return self.workdir / "ts_state"
