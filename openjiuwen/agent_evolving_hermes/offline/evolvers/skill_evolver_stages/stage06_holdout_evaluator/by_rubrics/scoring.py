from typing import Dict, Tuple

from .adaptive_rubric_weights import AdaptiveRubricWeights
from .judge import RubricsLLMJudge
from .score import RubricsFitnessScore


def run_rubrics_evaluation(module, holdout, config, console, label: str)\
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

    composite_mean = sum(composites) / len(composites) if composites else 0.0
    dim_means = {d: sum(dim_accum[d]) / len(dim_accum[d]) if dim_accum[d] else 0.0 for d in dim_names}

    b_list = [dim_means[d] for d in RubricsFitnessScore.DIM_NAMES]
    rubrics_score = AdaptiveRubricWeights().aggregate(b_list)

    console.print(f"  Pre-train holdout score (rubric): {rubrics_score:.4f}  ({n_holdout} examples)")
    return rubrics_score, dim_means
