# coding: utf-8
"""DemoConfig — shared run configuration loaded from config.json."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

# All valid run-mode identifiers, in canonical order.
ALL_MODES: List[str] = ["no_ts", "l2_only", "l3_only", "l2_l3", "no_ts_multi"]


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
        ``"no_ts"``, ``"l2_only"``, ``"l3_only"``, ``"l2_l3"``, ``"no_ts_multi"``.
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
        (used by ``l2_only`` and ``l2_l3`` modes).
    n_runs:
        How many independent GEPA runs to execute per mode.  With
        ``n_runs=1`` (default) the comparison table shows single values.
        With ``n_runs≥2`` it shows mean ± std and a bootstrap 95% CI
        so you can see whether one mode is reliably better or just got
        lucky on a single run.
    verbose:
        ``True`` → show full DSPy INFO training logs.
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
