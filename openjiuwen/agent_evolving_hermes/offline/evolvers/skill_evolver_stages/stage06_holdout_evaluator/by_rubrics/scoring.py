from __future__ import annotations

from typing import Dict, List, Tuple

from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from .judge import RubricsLLMJudge


def eval_rubrics_pass(
    module: SkillModule,
    holdout: list,
    multi_judge: RubricsLLMJudge,
    dim_names: List[str],
    label: str,
    console,
) -> Tuple[float, Dict[str, float]]:
    """Score *module* on *holdout* with the multi-objective judge."""
    n = len(holdout)
    dim_accum: Dict[str, List[float]] = {d: [] for d in dim_names}
    composites: List[float] = []

    for i, ex in enumerate(holdout, start=1):
        try:
            pred = module(task_input=ex.task_input)
            fs = multi_judge.score(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
                agent_output=getattr(pred, "output", ""),
                skill_text=module._skill_text_value,
            )
            vals = fs.as_list()
            for d, v in zip(dim_names, vals):
                dim_accum[d].append(v)
            composite = sum(vals) / len(vals)
            composites.append(composite)
            dims_str = ", ".join([f"{d}: {v:.2f}" for d, v in zip(dim_names, vals)])
            console.print(f"  [{i}/{n}] {label} → raw {composite:.4f} | ({dims_str})")
        except Exception:
            for d in dim_names:
                dim_accum[d].append(0.0)
            composites.append(0.0)
            console.print(f"  [{i}/{n}] {label} → 0.0000 (error)")

    composite_mean = sum(composites) / len(composites) if composites else 0.0
    dim_means = {
        d: sum(dim_accum[d]) / len(dim_accum[d]) if dim_accum[d] else 0.0
        for d in dim_names
    }
    return composite_mean, dim_means
