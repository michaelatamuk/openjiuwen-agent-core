from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from typing import Tuple

from .by_rubrics.scoring import run_rubrics_evaluation, evaluate_rubrics_path
from .holistic.scoring import run_holistic_evaluation, evaluate_holistic_path


def evaluate_on_holdout(optimized_module: SkillModule, scoring_mode: str, dataset,
                        config, console,
                        prior_metrics=None, prior_baseline_score_holistic=None,
                        prior_baseline_dims_rubrics=None,
                        raw_prebuilt_skill=None,
                        evolved_text=None) -> Tuple:
        console.print("\n[blue]~~~ Evolving Stage 06 - Evaluation On Holdout Started ~~~[/blue]")
        holdout = dataset.holdout or dataset.val

        if scoring_mode != "rubrics":
            result = evaluate_holistic_path(optimized_module, holdout, config, console,
                                            prior_metrics, prior_baseline_score_holistic)
        else:
            result = evaluate_rubrics_path(optimized_module, holdout, config, console,
                                           prior_baseline_dims_rubrics, raw_prebuilt_skill, evolved_text)

        console.print("\n[blue]~~~ Evolving Stage 06 - Evaluation On Holdout Finishe ~~~[/blue]")
        return result





def evaluate_baseline_on_holdout(baseline_module, dataset, config, console, needs_rubrics=False):
    holdout = dataset.holdout or dataset.val

    # 1. Run Holistic
    holistic_score = run_holistic_evaluation(baseline_module, holdout, config, console, "pre-train (single)")
    if not needs_rubrics:
        return holistic_score, None, None

    # 2. Run Rubrics
    rubrics_score, rubrics_dims = run_rubrics_evaluation(baseline_module, holdout, config, console, "pre-train skill")

    return holistic_score, rubrics_score, rubrics_dims
