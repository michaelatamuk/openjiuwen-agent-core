from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from typing import Dict, Tuple

from .adaptive_rubric_weights import AdaptiveRubricWeights
from .judge import RubricsLLMJudge
from .score import RubricsFitnessScore


def run_rubrics_evaluation(module: SkillModule, holdout, config, console, label: str)\
        -> Tuple[float, Dict[str, float]]:
    n_holdout = len(holdout)
    console.print(f"\n  [bold]Evaluating skill on holdout…[/bold] [dim]({n_holdout} examples, rubrics)[/dim]")

    judge = RubricsLLMJudge(model=config.eval_model)
    dim_names = RubricsFitnessScore.DIM_NAMES
    n = len(holdout)
    dim_accum = {d: [] for d in dim_names}
    composites = []

    for i, ex in enumerate(holdout, start=1):
        try:
            pred = module(task_input=ex.task_input)
            fs = judge.score(task_input=ex.task_input,
                                   expected_behavior=ex.expected_behavior,
                                   agent_output=getattr(pred, "output", ""),
                                   skill_text=module._skill_text_value)
            vals = fs.as_list()
            for d, v in zip(dim_names, vals):
                dim_accum[d].append(v)

            composite = sum(vals) / len(vals)
            composites.append(composite)

            # RESTORED: Detailed dimensional breakdown print
            dims_str = ", ".join([f"{d}: {v:.2f}" for d, v in zip(dim_names, vals)])
            console.print(f"  [{i}/{n}] {label} → raw {composite:.4f} | ({dims_str})")

        except Exception:
            for d in dim_names: dim_accum[d].append(0.0)
            composites.append(0.0)
            # RESTORED: Error logging
            console.print(f"  [{i}/{n}] {label} → 0.0000 (error)")

    dim_means = {d: sum(dim_accum[d]) / len(dim_accum[d]) if dim_accum[d] else 0.0 for d in dim_names}

    b_list = [dim_means[d] for d in RubricsFitnessScore.DIM_NAMES]
    rubrics_score = AdaptiveRubricWeights().aggregate(b_list)

    console.print(f"  Holdout score ({label}, unweighted): {rubrics_score:.4f}  ({n_holdout} examples)")
    return rubrics_score, dim_means


def evaluate_rubrics_path(module: SkillModule, holdout, config, console, baseline_dims, raw_skill, evolved_text) -> Tuple:
    if baseline_dims is None:
        raise ValueError(
            "scoring_mode='rubrics' requires pre-computed baseline dimension scores "
            "(prior_baseline_dims_rubrics). Run evaluate_baseline_on_holdout() first "
            "and pass the returned rubrics_dims to SkillEvolverParams."
        )

    evolved_score, evolved_dims = run_rubrics_evaluation(module, holdout, config, console, "evolved")

    # State management
    rubrics_state_path = config.output_dir / "rubrics_state.json"
    rubrics_state = AdaptiveRubricWeights.load_or_create(rubrics_state_path)

    b_list = [baseline_dims[d] for d in RubricsFitnessScore.DIM_NAMES]
    e_list = [evolved_dims[d] for d in RubricsFitnessScore.DIM_NAMES]

    # No-regression check
    nr_passed, failed_dims = rubrics_state.no_regression_passed(e_list, b_list)
    if not nr_passed:
        console.print(
            f"\n[red]No-regression check FAILED — {', '.join(failed_dims)} dropped > 0.02 vs baseline[/red]")
    else:
        console.print("\n[green]No-regression check ✓  all 5 dimensions passed[/green]")

    # Scoring and Penalty
    def _length_penalty(text: str) -> float:
        _len, _thr = len(text), config.max_skill_size * 0.90
        return min(0.30, 0.30 * (_len - _thr) / (config.max_skill_size - _thr)) if _len > _thr else 0.0

    final_e_score = rubrics_state.aggregate(e_list, length_penalty=_length_penalty(evolved_text))
    final_b_score = rubrics_state.aggregate(b_list, length_penalty=_length_penalty(raw_skill))

    console.print(
        f"  Weighted scores (stage 8b): baseline={final_b_score:.4f}  evolved={final_e_score:.4f}  Δ={final_e_score - final_b_score:+.4f}")

    rubrics_state.update_weights(e_list, b_list)
    rubrics_state.save(rubrics_state_path)

    return final_b_score, final_e_score, (final_e_score - final_b_score), None, evolved_dims, rubrics_state
