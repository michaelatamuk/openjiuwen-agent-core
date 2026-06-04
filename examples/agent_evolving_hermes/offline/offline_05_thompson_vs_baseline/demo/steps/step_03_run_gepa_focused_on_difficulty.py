
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, evolve_single_skill


def step(skills_root, skill_name, model, iterations, ts_batch_size, output_gepa_focused_on_difficulty, ts_state_dir,
         verbose: bool = False, baseline_score=None, run_index: int = 1, n_runs: int = 1,
         scoring_mode: str = "single"):
    _banner("③ GEPA — TS-TrainingSelector only (no Acceptance Gate)", run_index=run_index, n_runs=n_runs)
    print(f"  TS-TrainingSelector : selects top {ts_batch_size} of "
          f"10 train examples per iteration")
    print("    TS learns which examples best distinguish good vs bad evolved skills")
    print("    → discriminating examples (medium difficulty) accumulate higher α")
    print()
    print("  TS-AcceptanceGate   : OFF (threshold only, improvement ≥ 0.0)")
    print()

    evolver_config = EvolverConfig(
        skills_root = skills_root,
        output_dir = output_gepa_focused_on_difficulty,
        iterations = iterations,
        optimizer_model = model,
        eval_model = model,
        max_prompt_growth=0.5,
        verbose=verbose,
        # TS-TrainingSelector ON, TS-AcceptanceGate OFF
        ts_skill_scheduler = False,
        ts_example_selector = True,
        ts_example_batch_size = ts_batch_size,
        ts_acceptance_gate = False,
        ts_state_dir = ts_state_dir,
        scoring_mode=scoring_mode,
    )
    metrics_gepa_focused = evolve_single_skill(
        skill_name, "golden", config=evolver_config, min_improvement=0.0,
        prior_baseline_score_single=baseline_score,
    )

    if verbose:
        evolved_l2 = _read_latest_evolved(output_gepa_focused_on_difficulty, skill_name)
        _print_skill("  Evolved skill (TS-TrainingSelector only)", evolved_l2 or "[not produced]")

    return metrics_gepa_focused
