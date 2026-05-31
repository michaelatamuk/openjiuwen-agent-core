
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


def step(skills_root, SKILL_NAME, MODEL, ITERATIONS, TS_BATCH_SIZE, GOLDEN_EXAMPLES, output_ts, ts_state_dir):
    _banner("③ GEPA — with Thompson Sampling (Level 2 + Level 3)")
    print(f"  Level 2 — Example Selector  : selects top {TS_BATCH_SIZE} of "
          f"{int(len(GOLDEN_EXAMPLES)*0.5)} train examples per iteration")
    print("    TS learns which examples best distinguish good vs bad evolved skills")
    print("    → hard examples (security, concurrency) accumulate higher α")
    print()
    print(f"  Level 3 — Acceptance Gate   : P(candidate > deployed) ≥ 0.75")
    print("    Monte Carlo (100 draws) — prevents accepting a lucky one-off run")
    print()

    config_ts = EvolverConfig(
        skills_root  = skills_root,
        output_dir   = output_ts,
        iterations   = ITERATIONS,
        optimizer_model = MODEL,
        eval_model      = MODEL,
        # Thompson Sampling — Level 2 + Level 3 ON
        ts_skill_scheduler   = False,          # L1 only matters for --all runs
        ts_example_selector  = True,
        ts_example_batch_size = TS_BATCH_SIZE,
        ts_acceptance_gate   = True,
        ts_acceptance_confidence = 0.75,
        ts_acceptance_n_samples  = 100,
        ts_state_dir = ts_state_dir,
    )
    metrics_ts = evolve_single_skill(
        SKILL_NAME, "golden", config=config_ts, min_improvement=0.0
    )
    _print_metrics(metrics_ts)
    _print_ts_insights(ts_state_dir, SKILL_NAME)

    evolved_ts = _read_latest_evolved(output_ts, SKILL_NAME)
    _print_skill("  Evolved skill (with TS)", evolved_ts or "[not produced]")

    return metrics_ts
