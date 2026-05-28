from __future__ import annotations

import time
from typing import List, Tuple

import dspy

from openjiuwen.agent_evolving_hermess.offline.config import EvolverConfig
from openjiuwen.agent_evolving_hermess.offline.skills import SkillModule


def run_gepa_optimization(
    baseline_module: SkillModule,
    trainset: List,
    valset: List,
    config: EvolverConfig,
    console,
) -> Tuple[SkillModule, str, float]:
    """Run GEPA (or MIPROv2 fallback) to optimise the skill module.

    Returns (optimized_module, optimizer_name, elapsed_seconds).
    """
    console.print(f"\n[bold]Running GEPA[/bold] ({config.iterations} iterations)…")
    t0 = time.time()
    optimizer_name = "GEPA"

    try:
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric,
            max_steps=config.iterations,
        )
        optimized_module = optimizer.compile(
            baseline_module, trainset=trainset, valset=valset
        )
    except Exception as gepa_err:
        console.print(
            f"[yellow]GEPA not available ({gepa_err}), falling back to MIPROv2[/yellow]"
        )
        optimizer_name = "MIPROv2"
        optimizer = dspy.MIPROv2(metric=skill_fitness_metric, auto="light")
        optimized_module = optimizer.compile(
            baseline_module, trainset=trainset, valset=valset
        )

    elapsed = time.time() - t0
    console.print(f"[green]✓ Optimisation complete in {elapsed:.1f}s[/green]")
    return optimized_module, optimizer_name, elapsed


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Fast keyword-overlap metric for GEPA's inner optimisation loop.

    This is the function passed to dspy.GEPA(metric=...).
    Full LLM-as-judge is too expensive to run on every candidate.

    Mirrors Hermess skill_fitness_metric() exactly.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""

    if not agent_output.strip():
        return 0.0

    # Base score for non-empty output
    score = 0.5
    expected_words = set(expected.lower().split())
    output_words = set(agent_output.lower().split())

    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
        score = 0.3 + (0.7 * overlap)

    return min(1.0, max(0.0, score))
