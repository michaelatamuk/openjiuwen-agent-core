from typing import Tuple

from .by_rubrics.score import RubricsFitnessScore
from .by_rubrics.adaptive_rubric_weights import AdaptiveRubricWeights
from .by_rubrics.scoring import run_rubrics_evaluation
from .holistic.scoring import run_holistic_evaluation


def evaluate_on_holdout(optimized_module, scoring_mode: str, dataset,
                        config, console,
                        prior_metrics=None, prior_baseline_score_holistic=None,
                        prior_baseline_dims_rubrics=None,
                        raw_prebuilt_skill=None,
                        evolved_text=None) -> Tuple:
    console.print("\n[blue]~~~ Evolving Stage 06 - Evaluation On Holdout Started ~~~[/blue]")
    holdout = dataset.holdout or dataset.val

    if scoring_mode != "rubrics":
        baseline_score = prior_baseline_score_holistic
        evolved_score = run_holistic_evaluation(optimized_module, holdout, config, console, "evolved")

        improvement = evolved_score - prior_baseline_score_holistic
        delta_vs_prior = round(evolved_score - prior_metrics["evolved_score"], 4) if (
                    prior_metrics and "evolved_score" in prior_metrics) else None

        return baseline_score, evolved_score, improvement, delta_vs_prior, None, None

    # Rubrics Path
    evolved_score, evolved_dims = run_rubrics_evaluation(optimized_module, holdout, config, console, "evolved")

    rubrics_state_path = config.output_dir / "rubrics_state.json"
    rubrics_state = AdaptiveRubricWeights.load_or_create(rubrics_state_path)
    baseline_dims_list = [prior_baseline_dims_rubrics[d] for d in RubricsFitnessScore.DIM_NAMES]
    evolved_dims_list = [evolved_dims[d] for d in RubricsFitnessScore.DIM_NAMES]

    nr_passed, failed_dims = rubrics_state.no_regression_passed(evolved_dims_list, baseline_dims_list)
    if not nr_passed:
        console.print(f"\n[red]No-regression check FAILED — "
                      f"{', '.join(failed_dims)} dropped > 0.02 vs baseline[/red]")
    else:
        console.print("\n[green]No-regression check ✓  all 5 dimensions passed[/green]")

    def _length_penalty(text: str) -> float:
        _len = len(text)
        _thr = config.max_skill_size * 0.90
        if _len <= _thr:
            return 0.0
        return min(0.30, 0.30 * (_len - _thr) / (config.max_skill_size - _thr))

    evolved_score = rubrics_state.aggregate(evolved_dims_list, length_penalty=_length_penalty(evolved_text))
    baseline_score = rubrics_state.aggregate(baseline_dims_list, length_penalty=_length_penalty(raw_prebuilt_skill))
    improvement = evolved_score - baseline_score
    console.print(
        f"  Weighted scores (stage 8b): baseline={baseline_score:.4f}  evolved={evolved_score:.4f}"
        f"  Δ={improvement:+.4f}"
    )

    rubrics_state.update_weights(evolved_dims_list, baseline_dims_list)
    rubrics_state.save(rubrics_state_path)

    return baseline_score, evolved_score, evolved_score - baseline_score, None, evolved_dims, rubrics_state


def evaluate_baseline_on_holdout(baseline_module, dataset, config, console, needs_rubrics=False):
    holdout = dataset.holdout or dataset.val

    # 1. Run Holistic
    holistic_score = run_holistic_evaluation(baseline_module, holdout, config, console, "pre-train (single)")
    if not needs_rubrics:
        return holistic_score, None, None

    # 2. Run Rubrics
    rubrics_score, rubrics_dims = run_rubrics_evaluation(baseline_module, holdout, config, console, "pre-train skill")

    return holistic_score, rubrics_score, rubrics_dims
