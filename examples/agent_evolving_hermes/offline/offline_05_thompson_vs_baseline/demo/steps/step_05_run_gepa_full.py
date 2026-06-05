
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_ts_insights import \
    _print_ts_insights
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, evolve_single_skill


def step(skills_root, skill_name, model, itrations, ts_batch_size, examples, output_ts, ts_state_dir,
         verbose: bool = False, baseline_score=None, run_index: int = 1, n_runs: int = 1,
         scoring_mode: str = "single", baseline_score_multi=None, baseline_dims_multi=None):
    _banner("⑤ GEPA — with Thompson Sampling (TS-TrainingSelector + TS-AcceptanceGate)", run_index=run_index, n_runs=n_runs)
    print(f"  TS-TrainingSelector : selects top {ts_batch_size} of "
          f"{int(len(examples)*0.5)} train examples per iteration")
    print("    TS learns which examples best distinguish good vs bad evolved skills")
    print("    → discriminating examples (medium difficulty) accumulate higher α")
    print()
    print(f"  TS-AcceptanceGate   : P(candidate > deployed) ≥ 0.75")
    print("    Monte Carlo (100 draws) — prevents accepting a lucky one-off run")
    print()

    evolver_config = EvolverConfig(
        skills_root = skills_root,
        output_dir = output_ts,
        iterations = itrations,
        optimizer_model = model,
        eval_model = model,
        max_prompt_growth=20.0,  # allow up to 2000% growth; GEPA now rewrites the full skill body
        verbose=verbose,
        # TS-TrainingSelector + TS-AcceptanceGate ON
        ts_skill_scheduler = False,          # skill scheduler only matters for --all runs
        ts_example_selector = True,
        ts_example_batch_size = ts_batch_size,
        ts_acceptance_gate = True,
        ts_acceptance_confidence = 0.75,
        ts_acceptance_n_samples = 100,
        ts_state_dir = ts_state_dir,
        scoring_mode=scoring_mode,
    )
    metrics_ts = evolve_single_skill(
        skill_name, "golden", config=evolver_config, min_improvement=0.0,
        prior_baseline_score_single=baseline_score,
        prior_baseline_score_multi=baseline_score_multi,
        prior_baseline_dims_multi=baseline_dims_multi,
    )
    _print_ts_insights(ts_state_dir, skill_name)

    if verbose:
        evolved_ts = _read_latest_evolved(output_ts, skill_name)
        _print_skill("  Evolved skill (with TS)", evolved_ts or "[not produced]")

    return metrics_ts
