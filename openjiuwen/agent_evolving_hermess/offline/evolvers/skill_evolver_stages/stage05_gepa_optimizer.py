from __future__ import annotations

import time
from typing import List, Tuple

import dspy

from openjiuwen.agent_evolving_hermess.offline.config import EvolverConfig
from openjiuwen.agent_evolving_hermess.offline.fitness import skill_fitness_metric
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
