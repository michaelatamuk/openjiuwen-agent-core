# coding: utf-8
"""Evaluate the baseline skill on the holdout set WITHOUT any GEPA training."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.steps_shared_object import \
    SharedEvolutionObjects
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage08_holdout_evaluator import (
    evaluate_baseline_on_holdout,
)

# Modes that require a multi-objective baseline pre-evaluation.
_MULTI_MODES = {"gepa_rubric"}


def run_step(shared_evolution_object: SharedEvolutionObjects,
             skills_root: Path,
             model: str,
             output_dir: Path,
             run_modes: Optional[List[str]],
             console,
             verbose: bool = False,):
    """Score the baseline skill on holdout for every scoring system that will be used.

    Evaluates the single-score baseline unconditionally.  If any mode in
    *run_modes* requires multi-objective scoring (currently ``"gepa_rubric"``),
    the 5-dimension baseline is also evaluated here so that GEPA runs never
    need to re-evaluate the baseline themselves.

    Parameters
    ----------
    shared_evolution_object:
        Built by step_01 — contains baseline_module and dataset.
    skills_root:
        Root directory that contains the skill sub-directory.
    model:
        DSPy model string used for both running the skill and judging.
    output_dir:
        Directory where the golden dataset cache is written.
    run_modes:
        List of mode names from config.  Drives whether multi-objective
        baseline evaluation is performed.  Pass ``None`` / ``[]`` to
        skip multi evaluation.
    verbose:
        ``True`` → show DSPy / Rich INFO logs during evaluation.

    Returns
    -------
    (single_score, multi_score, multi_dims)
        ``single_score``  — composite holdout score in [0, 1] (always computed).
        ``multi_score``   — equal-weight composite from the multi-objective
                            judge, or ``None`` when no multi mode was requested.
        ``multi_dims``    — ``{dim: mean_score}`` from the multi-objective
                            judge, or ``None`` when no multi mode was requested.
    """
    console.print(f"\n[bold cyan]*** Demo Step 02: Evaluate Baseline Started ***[/bold cyan]")

    needs_multi = bool(run_modes and _MULTI_MODES.intersection(run_modes))
    modes_label = "single + multi" if needs_multi else "single"
    _banner(f"① PRE-TRAINING — holdout evaluation ({modes_label})", console=console)
    output_dir.mkdir(parents=True, exist_ok=True)

    evolver_config = EvolverConfig(skills_root=skills_root,
                                   output_dir=output_dir,
                                   optimizer_model=model,
                                   eval_model=model,
                                   verbose=verbose)

    single_score, multi_score, multi_dims = evaluate_baseline_on_holdout(
        shared_evolution_object.baseline_module,
        shared_evolution_object.dataset,
        evolver_config,
        console,
        needs_multi=needs_multi,
    )

    console.print(f"[bold cyan]*** Demo Step 02: Evaluate Baseline Finished ***[/bold cyan]")
    return single_score, multi_score, multi_dims
