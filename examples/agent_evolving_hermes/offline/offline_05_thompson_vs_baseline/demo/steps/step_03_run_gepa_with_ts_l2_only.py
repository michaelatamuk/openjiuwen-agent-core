
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, evolve_single_skill


def step(skills_root, skill_name, model, iterations, ts_batch_size, output_l2_only, ts_state_dir,
         verbose: bool = False, baseline_score=None, run_index: int = 1, n_runs: int = 1):
    _banner("③-L2 GEPA — Level 2 only (Example Selector, no Acceptance Gate)", run_index=run_index, n_runs=n_runs)
    print(f"  Level 2 — Example Selector  : selects top {ts_batch_size} of "
          f"10 train examples per iteration")
    print("    TS learns which examples best distinguish good vs bad evolved skills")
    print("    → hard examples (security, concurrency) accumulate higher α")
    print()
    print("  Level 3 — Acceptance Gate   : OFF (threshold only, improvement ≥ 0.0)")
    print()

    evolver_config = EvolverConfig(
        skills_root = skills_root,
        output_dir = output_l2_only,
        iterations = iterations,
        optimizer_model = model,
        eval_model = model,
        max_prompt_growth=0.5,
        verbose=verbose,
        # Thompson Sampling — Level 2 ON, Level 3 OFF
        ts_skill_scheduler = False,
        ts_example_selector = True,
        ts_example_batch_size = ts_batch_size,
        ts_acceptance_gate = False,
        ts_state_dir = ts_state_dir,
    )
    metrics_l2 = evolve_single_skill(
        skill_name, "golden", config=evolver_config, min_improvement=0.0,
        prior_baseline_score=baseline_score,
    )

    if verbose:
        evolved_l2 = _read_latest_evolved(output_l2_only, skill_name)
        _print_skill("  Evolved skill (L2 only)", evolved_l2 or "[not produced]")

    return metrics_l2
