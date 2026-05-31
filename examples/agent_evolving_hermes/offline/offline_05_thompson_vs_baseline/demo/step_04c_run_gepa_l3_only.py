
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_metrics import \
    _print_metrics
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_ts_insights import \
    _print_ts_insights
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved
from offline import EvolverConfig, evolve_single_skill


def step(skills_root, SKILL_NAME, MODEL, ITERATIONS, output_l3_only, ts_state_dir,
         verbose: bool = False):
    _banner("④-L3 GEPA — Level 3 only (Acceptance Gate, all examples equal weight)")
    print("  Level 2 — Example Selector  : OFF (all training examples, equal weight)")
    print()
    print(f"  Level 3 — Acceptance Gate   : P(candidate > deployed) ≥ 0.75")
    print("    Monte Carlo (100 draws) — prevents accepting a lucky one-off run")
    print()

    config_l3 = EvolverConfig(
        skills_root  = skills_root,
        output_dir   = output_l3_only,
        iterations   = ITERATIONS,
        optimizer_model = MODEL,
        eval_model      = MODEL,
        max_prompt_growth=0.5,
        verbose=verbose,
        # Thompson Sampling — Level 2 OFF, Level 3 ON
        ts_skill_scheduler       = False,
        ts_example_selector      = False,
        ts_acceptance_gate       = True,
        ts_acceptance_confidence = 0.75,
        ts_acceptance_n_samples  = 100,
        ts_state_dir             = ts_state_dir,
    )
    metrics_l3 = evolve_single_skill(
        SKILL_NAME, "golden", config=config_l3, min_improvement=0.0
    )
    _print_metrics(metrics_l3)
    _print_ts_insights(ts_state_dir, SKILL_NAME)

    evolved_l3 = _read_latest_evolved(output_l3_only, SKILL_NAME)
    _print_skill("  Evolved skill (L3 only)", evolved_l3 or "[not produced]")

    return metrics_l3
