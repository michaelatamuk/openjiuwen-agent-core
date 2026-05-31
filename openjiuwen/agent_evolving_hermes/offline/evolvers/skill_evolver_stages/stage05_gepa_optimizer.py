from __future__ import annotations

import time
from typing import List, Tuple

import dspy

from openjiuwen.agent_evolving_hermes.offline.config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from openjiuwen.agent_evolving_hermes.offline.evolvers.selection import make_example_selector


def run_gepa_optimization(
    baseline_module: SkillModule,
    trainset: List,
    valset: List,
    config: EvolverConfig,
    console,
    skill_name: str = "unknown",
) -> Tuple[SkillModule, str, float]:
    """Run GEPA (or MIPROv2 fallback) to optimise the skill module.

    When ``config.ts_example_selector`` is True, the training examples are
    chosen by a Level 2 Thompson Sampling selector instead of using the full
    set.  After GEPA completes the per-example fitness scores are fed back to
    update the selector's Beta arms so future runs concentrate on the most
    discriminating examples.

    Returns (optimized_module, optimizer_name, elapsed_seconds).
    """
    # ── Level 2: select training examples via factory ─────────────────────────
    selector = make_example_selector(trainset, skill_name, config)
    selected_trainset = selector.select()

    ts_active = getattr(config, "ts_example_selector", False)
    if ts_active and len(selected_trainset) < len(trainset):
        console.print(
            f"\n[bold]Running GEPA[/bold] ({config.iterations} iterations)… "
            f"[dim][TS: {len(selected_trainset)}/{len(trainset)} examples][/dim]"
        )
    else:
        console.print(f"\n[bold]Running GEPA[/bold] ({config.iterations} iterations)…")

    t0 = time.time()
    optimizer_name = "GEPA"

    try:
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric,
            max_steps=config.iterations,
        )
        optimized_module = optimizer.compile(
            baseline_module, trainset=selected_trainset, valset=valset
        )
    except Exception as gepa_err:
        console.print(
            f"[yellow]GEPA not available ({gepa_err}), falling back to MIPROv2[/yellow]"
        )
        optimizer_name = "MIPROv2"
        optimizer = dspy.MIPROv2(metric=skill_fitness_metric, auto="light")
        optimized_module = optimizer.compile(
            baseline_module, trainset=selected_trainset, valset=valset
        )

    elapsed = time.time() - t0
    console.print(f"[green]✓ Optimisation complete in {elapsed:.1f}s[/green]")

    # ── Update example selector arms with per-example fitness ─────────────────
    if ts_active:
        fitnesses: List[float] = []
        for ex in selected_trainset:
            try:
                pred = optimized_module(task_input=ex.task_input)
                f = skill_fitness_metric(ex, pred)
            except Exception:
                f = 0.0
            fitnesses.append(f)
        selector.update(selected_trainset, fitnesses)

    return optimized_module, optimizer_name, elapsed


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Fast keyword-overlap metric for GEPA's inner optimisation loop.

    This is the function passed to dspy.GEPA(metric=...).
    Full LLM-as-judge is too expensive to run on every candidate.

    Mirrors Hermes skill_fitness_metric() exactly.
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
