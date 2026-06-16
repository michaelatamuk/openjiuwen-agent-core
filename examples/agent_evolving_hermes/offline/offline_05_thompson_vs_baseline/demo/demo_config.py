# coding: utf-8
"""DemoConfig — shared run configuration loaded from config.json."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# All valid run-mode identifiers, in canonical order.
ALL_MODES: List[str] = [
    "gepa_plain_holistic",
    "gepa_plain_rubrics",
    "gepa_plain_graph",
    "gepa_plain_checklist",
    "gepa_plain_instruction_following",
    "gepa_plain_consistency",
    "gepa_plain_comparative",
    "gepa_scoring_matrix",
    "gepa_focused_on_difficulty",
    "gepa_gated",
    "gepa_full",
]


@dataclass
class DemoConfig:
    """Shared settings that apply to every scenario in a run.

    Loaded from ``config.json`` by :func:`demo.config_loader.load_config`.
    All scenario-specific data (skill text, golden examples, workdir)
    lives in :class:`~demo.demo_params.DemoParams` instead.

    Attributes
    ----------
    scenario_names:
        Which scenarios to run, in order.  Each name must match a key
        returned by :func:`scenarios.scenario.list_scenarios`.
    run_modes:
        Which GEPA training passes to execute.  Valid values:
        ``" "gepa_plain_holistic""``, ''"gepa_plain_rubrics", ``"gepa_focused_on_difficulty"``, ``"gepa_gated"``, ``"gepa_full"````.
        Pass ``[]`` for a baseline-only evaluation (no GEPA training).
    api_key:
        API key for the LLM provider (DeepSeek by default).
        Overridden by the ``DEEPSEEK_API_KEY`` environment variable.
    model:
        DSPy model string, e.g. ``"deepseek/deepseek-chat"``.
    api_base:
        Base URL of the LLM provider API.
    iterations:
        Number of GEPA optimisation iterations per training pass.
    ts_batch_size:
        Number of examples the TS example-selector picks per batch
        (used by ``gepa_focused_on_difficulty`` and ``gepa_full`` modes).
    n_runs:
        How many independent GEPA runs to execute per mode.  With
        ``n_runs=1`` (default) the comparison table shows single values.
        With ``n_runs≥2`` it shows mean ± std and a bootstrap 95% CI
        so you can see whether one mode is reliably better or just got
        lucky on a single run.
    verbose:
        ``True`` → show full DSPy INFO training logs.
    print_skill_diff:
        ``True`` → after the comparison table, print the baseline skill and
        the winner skill side by side so changes are easy to read.
    fitness_metrics:
        One or more fast proxy metrics used inside the GEPA optimizer loop.
        Built-ins: ``"f1"`` (stop-word-filtered F1, default),
        ``"bag_of_words"`` (word-bag with 0.3 floor),
        ``"graph"`` (concept-graph structural similarity).
        Custom: dotted import path (e.g.
        ``"examples.agent_evolving_hermes.offline.custom_fitness_metric_tech_keywords.tech_keyword_fitness_metric"``).
        When multiple values are given each mode runs once per metric, producing
        independent output directories (e.g. ``output_gepa_plain_holistic__jiuwen``).
    oracle_data_dir:
        Persistent directory where every ``scoring_matrix_*.json`` produced by
        ``gepa_scoring_matrix`` mode is also copied.  All skills accumulate here
        across runs, enabling ``build_oracle_dataset.py`` to aggregate them into
        a single oracle training table.  Supports ``~`` expansion.
        ``None`` (default) disables the copy — matrices are only saved in the
        per-run temp workdir.
    """

    scenario_names: List[str]
    run_modes: List[str]
    api_key: str
    model: str
    api_base: str = "https://api.deepseek.com"
    iterations: int = 5
    ts_batch_size: int = 4
    n_runs: int = 1
    verbose: bool = False
    print_skill_diff: bool = False
    fitness_metrics: List[str] = field(default_factory=lambda: ["bag_of_words"])
    oracle_data_dir: Optional[str] = None
