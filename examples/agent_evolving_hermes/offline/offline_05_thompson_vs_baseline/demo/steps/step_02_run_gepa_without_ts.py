
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_metrics import \
    _print_metrics
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved
from offline import EvolverConfig, evolve_single_skill


def step(skills_root, skill_name, model, iterations, output_no_ts, verbose: bool = False):
    _banner("② GEPA — without Thompson Sampling")
    print("  Example selector : all training examples, equal weight")
    print("  Acceptance gate  : threshold only (improvement ≥ 0.0)")
    print()

    config_no_ts = EvolverConfig(
        skills_root=skills_root,
        output_dir=output_no_ts,
        iterations=iterations,
        optimizer_model=model,
        eval_model=model,
        max_prompt_growth=0.5,   # allow up to 50% growth; baseline skill is intentionally short
        verbose=verbose,
        # Thompson Sampling — all OFF
        ts_skill_scheduler=False,
        ts_example_selector=False,
        ts_acceptance_gate=False,
    )
    metrics_no_ts = evolve_single_skill(
        skill_name, "golden", config=config_no_ts, min_improvement=0.0
    )
    _print_metrics(metrics_no_ts)

    if verbose:
        evolved_no_ts = _read_latest_evolved(output_no_ts, skill_name)
        _print_skill("  Evolved skill (no TS)", evolved_no_ts or "[not produced]")

    return metrics_no_ts
