from typing import Tuple
from .holistic.scoring import run_holistic_evaluation
from .by_rubrics.scoring import run_rubrics_evaluation


def evaluate_on_holdout(baseline_module, optimized_module, dataset, config, console,
                        prior_metrics=None, prior_baseline_score_holistic=None,
                        scoring_mode="holistic", prior_baseline_score_rubrics=None) -> Tuple:
    console.print("\n[blue]~~~ Evolving Stage 06 - Evaluation On Holdout Started ~~~[/blue]")
    holdout = dataset.holdout or dataset.val

    if scoring_mode != "rubrics":
        b_score = prior_baseline_score_holistic or run_holistic_evaluation(baseline_module, holdout, config, console,
                                                                           "pre-train")
        e_score = run_holistic_evaluation(optimized_module, holdout, config, console, "evolved")

        imp = e_score - b_score
        delta = round(e_score - prior_metrics["evolved_score"], 4) if (
                    prior_metrics and "evolved_score" in prior_metrics) else None
        return b_score, e_score, imp, delta, None

    # Rubrics Path
    b_score = prior_baseline_score_rubrics or \
              run_rubrics_evaluation(baseline_module, holdout, config, console, "pre-train")[0]
    e_comp, e_dims = run_rubrics_evaluation(optimized_module, holdout, config, console, "evolved")

    return b_score, e_comp, e_comp - b_score, None, e_dims


def evaluate_baseline_on_holdout(baseline_module, dataset, config, console, needs_rubrics=False):
    holdout = dataset.holdout or dataset.val

    # 1. Run Holistic
    holistic_score = run_holistic_evaluation(baseline_module, holdout, config, console, "pre-train (single)")
    if not needs_rubrics:
        return holistic_score, None, None

    # 2. Run Rubrics
    rubrics_score, rubrics_dims = run_rubrics_evaluation(baseline_module, holdout, config, console, "pre-train skill")

    return holistic_score, rubrics_score, rubrics_dims
