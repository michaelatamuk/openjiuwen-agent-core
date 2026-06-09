from __future__ import annotations

import time
from typing import List

import dspy

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from openjiuwen.agent_evolving_hermes.offline.evolvers.selection import make_example_selector
from ._evolved_skill_extractor import extract_evolved_skill
from ._fitness_metric_resolver import resolve_fitness_metric


def run_gepa_optimization(baseline_module: SkillModule,
                          skill_frontmatter_text,
                          trainset: List,
                          valset: List,
                          config: EvolverConfig,
                          console,
                          skill_name: str = "unknown"):
    """Run GEPA (or MIPROv2 fallback) to optimise the skill module.

    When ``config.ts_example_selector`` is True, the training examples are
    chosen by a Level 2 Thompson Sampling selector instead of using the full
    set.  After GEPA completes the per-example fitness scores are fed back to
    update the selector's Beta arms so future runs concentrate on the most
    discriminating examples.

    Returns (optimized_module, optimizer_name, elapsed_seconds).
    """
    console.print("\n[blue]~~~ Evolving Stage 05 - GEPA Optimization Run Started ~~~[/blue]")

    # ── Resolve fitness metric callable ───────────────────────────────────────
    fitness_metric_fn = resolve_fitness_metric(
        getattr(config, "fitness_metric", "jiuwen"),
        getattr(config, "custom_fitness_metrics", {}),
    )

    # ── Level 2: select training examples via factory ─────────────────────────
    examples_selector = make_example_selector(trainset, skill_name, config)
    selected_trainset_examples = examples_selector.select()

    console.print(f"\n[bold]Running GEPA[/bold] ({config.iterations} iterations)…", end="")

    ts_active = getattr(config, "ts_example_selector", False)
    if ts_active and len(selected_trainset_examples) < len(trainset):
        console.print(f" [dim][TS: {len(selected_trainset_examples)}/{len(trainset)} examples][/dim]")

    metric_name = getattr(config, "fitness_metric", "jiuwen")
    console.print(f"\n[dim]  ↳ ~{config.iterations + 2} candidate skills will be scored below;"
                  f" each 'Average Metric: X / N (Y%)' line = {metric_name} score (not LLM-judge)[/dim]")

    t0 = time.time()
    optimizer_name = "GEPA"

    try:
        optimizer = dspy.GEPA(metric=fitness_metric_fn,
                              max_full_evals=config.iterations,
                              reflection_lm=dspy.settings.lm)
    except Exception as gepa_err:
        console.print(f"[yellow]GEPA not available ({gepa_err}), falling back to MIPROv2[/yellow]")
        optimizer_name = "MIPROv2"
        optimizer = dspy.MIPROv2(metric=fitness_metric_fn, auto="light")

    optimized_module = optimizer.compile(baseline_module,
                                         trainset=selected_trainset_examples,
                                         valset=valset)

    elapsed = time.time() - t0
    console.print(f"[green]✓ Optimisation complete in {elapsed:.1f}s[/green]")

    # ── Update example selector arms with per-example fitness ─────────────────
    if ts_active:
        fitnesses: List[float] = []
        for ex in selected_trainset_examples:
            try:
                pred = optimized_module(task_input=ex.task_input)
                f = fitness_metric_fn(ex, pred)
            except Exception:
                f = 0.0
            fitnesses.append(f)
        examples_selector.update(selected_trainset_examples, fitnesses)

    evolved_text = extract_evolved_skill(optimized_module, skill_frontmatter_text, console)

    console.print("[blue]~~~ Evolving Stage 05 - GEPA Optimization Run Finished ~~~[/blue]")
    return optimized_module, optimizer_name, elapsed, evolved_text
