
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_metrics import \
    _print_metrics
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved
from offline import EvolverConfig, evolve_single_skill


def step(skills_root, SKILL_NAME, MODEL, ITERATIONS, TS_BATCH_SIZE, output_l2_only, ts_state_dir,
         verbose: bool = False):
    _banner("④-L2 GEPA — Level 2 only (Example Selector, no Acceptance Gate)")
    print(f"  Level 2 — Example Selector  : selects top {TS_BATCH_SIZE} of "
          f"10 train examples per iteration")
    print("    TS learns which examples best distinguish good vs bad evolved skills")
    print("    → hard examples (security, concurrency) accumulate higher α")
    print()
    print("  Level 3 — Acceptance Gate   : OFF (threshold only, improvement ≥ 0.0)")
    print()

    config_l2 = EvolverConfig(
        skills_root  = skills_root,
        output_dir   = output_l2_only,
        iterations   = ITERATIONS,
        optimizer_model = MODEL,
        eval_model      = MODEL,
        max_prompt_growth=0.5,
        verbose=verbose,
        # Thompson Sampling — Level 2 ON, Level 3 OFF
        ts_skill_scheduler    = False,
        ts_example_selector   = True,
        ts_example_batch_size = TS_BATCH_SIZE,
        ts_acceptance_gate    = False,
        ts_state_dir          = ts_state_dir,
    )
    metrics_l2 = evolve_single_skill(
        SKILL_NAME, "golden", config=config_l2, min_improvement=0.0
    )
    _print_metrics(metrics_l2)

    if verbose:
        evolved_l2 = _read_latest_evolved(output_l2_only, SKILL_NAME)
        _print_skill("  Evolved skill (L2 only)", evolved_l2 or "[not produced]")

    return metrics_l2
