from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from typing import Optional, Tuple

from .by_checklist.scoring import run_checklist_evaluation, evaluate_checklist_path
from .by_comparative.scoring import run_comparative_evaluation, evaluate_comparative_path
from .by_consistency.scoring import run_consistency_evaluation, evaluate_consistency_path
from .by_graph.scoring import run_graph_evaluation, evaluate_graph_path
from .by_instruction_following.scoring import (
    run_instruction_following_evaluation,
    evaluate_instruction_following_path,
)
from .by_rubrics.scoring import run_rubrics_evaluation, evaluate_rubrics_path
from .holistic.scoring import run_holistic_evaluation, evaluate_holistic_path


def evaluate_on_holdout(optimized_module: SkillModule, scoring_mode: str, dataset,
                        config, console,
                        prior_metrics=None,
                        prior_baseline_score_holistic=None,
                        prior_baseline_score_graph=None,
                        prior_baseline_dims_rubrics=None,
                        raw_prebuilt_skill=None,
                        evolved_text=None,
                        # ── new judge paths ──────────────────────────────────
                        prior_baseline_score_checklist=None,
                        prior_baseline_score_instruction_following=None,
                        prior_baseline_score_consistency=None,
                        baseline_module=None,         # required for scoring_mode="comparative"
                        ) -> Tuple:
    console.print("\n[blue]~~~ Evolving Stage 06 - Evaluation On Holdout Started ~~~[/blue]")
    holdout = dataset.holdout or dataset.val

    if scoring_mode == "rubrics":
        result = evaluate_rubrics_path(optimized_module, holdout, config, console,
                                       prior_baseline_dims_rubrics, raw_prebuilt_skill, evolved_text)
    elif scoring_mode == "graph":
        result = evaluate_graph_path(optimized_module, holdout, config, console,
                                     prior_metrics, prior_baseline_score_graph)
    elif scoring_mode == "checklist":
        result = evaluate_checklist_path(optimized_module, holdout, config, console,
                                         prior_metrics, prior_baseline_score_checklist)
    elif scoring_mode == "instruction_following":
        result = evaluate_instruction_following_path(optimized_module, holdout, config, console,
                                                     prior_metrics,
                                                     prior_baseline_score_instruction_following)
    elif scoring_mode == "consistency":
        result = evaluate_consistency_path(optimized_module, holdout, config, console,
                                           prior_metrics, prior_baseline_score_consistency)
    elif scoring_mode == "comparative":
        if baseline_module is None:
            raise ValueError(
                "scoring_mode='comparative' requires baseline_module to be provided. "
                "Pass params.prebuilt_baseline_module via evaluate_on_holdout()."
            )
        result = evaluate_comparative_path(baseline_module, optimized_module, holdout,
                                           config, console, prior_metrics)
    else:
        result = evaluate_holistic_path(optimized_module, holdout, config, console,
                                        prior_metrics, prior_baseline_score_holistic)

    console.print("\n[blue]~~~ Evolving Stage 06 - Evaluation On Holdout Finished ~~~[/blue]")
    return result


def evaluate_baseline_on_holdout(baseline_module, dataset, config, console,
                                  needs_rubrics=False,
                                  needs_graph=False,
                                  needs_checklist=False,
                                  needs_instruction_following=False,
                                  needs_consistency=False):
    """Evaluate the baseline module on holdout and return pre-computed scores.

    Returns
    -------
    (holistic_score, rubrics_score, rubrics_dims, graph_score,
     checklist_score, instruction_following_score, consistency_score)

    Fields are ``None`` when the corresponding ``needs_*`` flag is ``False``.
    Comparative does not require a pre-computed baseline (it always uses the
    fixed neutral score of 0.5) so there is no ``needs_comparative`` param.
    """
    holdout = dataset.holdout or dataset.val

    # 1. Always run holistic
    holistic_score = run_holistic_evaluation(
        baseline_module, holdout, config, console, "pre-train (holistic)"
    )

    # 2. Optionally run rubrics
    rubrics_score, rubrics_dims = None, None
    if needs_rubrics:
        rubrics_score, rubrics_dims = run_rubrics_evaluation(
            baseline_module, holdout, config, console, "pre-train (rubrics)"
        )

    # 3. Optionally run graph (algorithmic, no LLM)
    graph_score: Optional[float] = None
    if needs_graph:
        graph_score, _ = run_graph_evaluation(
            baseline_module, holdout, config, console, "pre-train (graph)"
        )

    # 4. Optionally run checklist
    checklist_score: Optional[float] = None
    if needs_checklist:
        checklist_score, _ = run_checklist_evaluation(
            baseline_module, holdout, config, console, "pre-train (checklist)"
        )

    # 5. Optionally run instruction-following
    instruction_following_score: Optional[float] = None
    if needs_instruction_following:
        instruction_following_score, _ = run_instruction_following_evaluation(
            baseline_module, holdout, config, console, "pre-train (instruction-following)"
        )

    # 6. Optionally run consistency (no LLM — multiple forward passes)
    consistency_score: Optional[float] = None
    if needs_consistency:
        consistency_score, _ = run_consistency_evaluation(
            baseline_module, holdout, config, console, "pre-train (consistency)"
        )

    return (holistic_score, rubrics_score, rubrics_dims, graph_score,
            checklist_score, instruction_following_score, consistency_score)
