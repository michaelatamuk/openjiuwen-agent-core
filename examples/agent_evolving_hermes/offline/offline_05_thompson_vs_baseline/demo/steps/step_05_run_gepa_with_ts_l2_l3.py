
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_ts_insights import \
    _print_ts_insights
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, evolve_single_skill


def step(skills_root, skill_name, model, itrations, ts_batch_size, examples, output_ts, ts_state_dir,
         verbose: bool = False, baseline_score=None, run_index: int = 1, n_runs: int = 1):
    _banner("⑤ GEPA — with Thompson Sampling (Level 2 + Level 3)", run_index=run_index, n_runs=n_runs)
    print(f"  Level 2 — Example Selector  : selects top {ts_batch_size} of "
          f"{int(len(examples)*0.5)} train examples per iteration")
    print("    TS learns which examples best distinguish good vs bad evolved skills")
    print("    → hard examples (security, concurrency) accumulate higher α")
    print()
    print(f"  Level 3 — Acceptance Gate   : P(candidate > deployed) ≥ 0.75")
    print("    Monte Carlo (100 draws) — prevents accepting a lucky one-off run")
    print()

    evolver_config = EvolverConfig(
        skills_root = skills_root,
        output_dir = output_ts,
        iterations = itrations,
        optimizer_model = model,
        eval_model = model,
        max_prompt_growth=0.5,   # allow up to 50% growth; baseline skill is intentionally short
        verbose=verbose,
        # Thompson Sampling — Level 2 + Level 3 ON
        ts_skill_scheduler = False,          # L1 only matters for --all runs
        ts_example_selector = True,
        ts_example_batch_size = ts_batch_size,
        ts_acceptance_gate = True,
        ts_acceptance_confidence = 0.75,
        ts_acceptance_n_samples = 100,
        ts_state_dir = ts_state_dir,
    )
    metrics_ts = evolve_single_skill(
        skill_name, "golden", config=evolver_config, min_improvement=0.0,
        prior_baseline_score=baseline_score,
    )
    _print_ts_insights(ts_state_dir, skill_name)

    if verbose:
        evolved_ts = _read_latest_evolved(output_ts, skill_name)
        _print_skill("  Evolved skill (with TS)", evolved_ts or "[not produced]")

    return metrics_ts
