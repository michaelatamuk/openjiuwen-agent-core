# coding: utf-8
"""Evaluate the baseline skill on the holdout set WITHOUT any GEPA training."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.steps_shared_object import \
    SharedEvolutionObjects
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage06_holdout_evaluator.holdout_evaluator import evaluate_baseline_on_holdout

# Modes that require specific baseline pre-evaluations.
_RUBRIC_MODES        = {"gepa_plain_rubrics"}
_GRAPH_MODES         = {"gepa_plain_graph"}
_CHECKLIST_MODES     = {"gepa_plain_checklist"}
_IF_MODES            = {"gepa_plain_instruction_following"}
_CONSISTENCY_MODES   = {"gepa_plain_consistency"}
# Comparative uses a fixed 0.5 neutral baseline — no pre-computation needed.


def run_step(shared_evolution_object: SharedEvolutionObjects,
             skills_root: Path,
             model: str,
             output_dir: Path,
             run_modes: Optional[List[str]],
             console,
             verbose: bool = False):
    """Score the baseline skill on holdout for every scoring system that will be used.

    Evaluates the holistic baseline unconditionally.  Additional evaluations
    are triggered only when a mode in *run_modes* requires them:

    * ``"gepa_plain_rubrics"``               → 5-dimension rubrics baseline
    * ``"gepa_plain_graph"``                 → graph similarity baseline (no LLM)
    * ``"gepa_plain_checklist"``             → checklist pass-rate baseline
    * ``"gepa_plain_instruction_following"`` → instruction-following baseline
    * ``"gepa_plain_consistency"``           → output consistency baseline (no judge)

    Parameters
    ----------
    shared_evolution_object:
        Built by step_01 — contains baseline_module and dataset.
    skills_root, model, output_dir, console, verbose:
        Standard demo step params.
    run_modes:
        List of mode names from config.  Drives which baseline evaluations run.
        Pass ``None`` / ``[]`` to skip non-holistic evaluations.

    Returns
    -------
    (holistic_score, rubrics_score, rubrics_dims, graph_score,
     checklist_score, instruction_following_score, consistency_score)
        Non-requested scores are ``None``.
    """
    console.print(f"\n[bold cyan]*** Demo Step 02: Evaluate Baseline Started ***[/bold cyan]")

    modes_set = set(run_modes) if run_modes else set()
    needs_rubrics               = bool(modes_set & _RUBRIC_MODES)
    needs_graph                 = bool(modes_set & _GRAPH_MODES)
    needs_checklist             = bool(modes_set & _CHECKLIST_MODES)
    needs_instruction_following = bool(modes_set & _IF_MODES)
    needs_consistency           = bool(modes_set & _CONSISTENCY_MODES)

    parts = ["holistic"]
    if needs_rubrics:               parts.append("rubrics")
    if needs_graph:                 parts.append("graph")
    if needs_checklist:             parts.append("checklist")
    if needs_instruction_following: parts.append("instruction-following")
    if needs_consistency:           parts.append("consistency")
    _banner(f"① PRE-TRAINING — holdout evaluation ({' + '.join(parts)})", console=console)
    output_dir.mkdir(parents=True, exist_ok=True)

    evolver_config = EvolverConfig(skills_root=skills_root,
                                   output_dir=output_dir,
                                   optimizer_model=model,
                                   eval_model=model,
                                   verbose=verbose)

    (single_score, rubrics_score, rubrics_dims, graph_score,
     checklist_score, instruction_following_score, consistency_score) = evaluate_baseline_on_holdout(
        shared_evolution_object.baseline_module,
        shared_evolution_object.dataset,
        evolver_config,
        console,
        needs_rubrics=needs_rubrics,
        needs_graph=needs_graph,
        needs_checklist=needs_checklist,
        needs_instruction_following=needs_instruction_following,
        needs_consistency=needs_consistency,
    )

    console.print(f"[bold cyan]*** Demo Step 02: Evaluate Baseline Finished ***[/bold cyan]")
    return (single_score, rubrics_score, rubrics_dims, graph_score,
            checklist_score, instruction_following_score, consistency_score)
