
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_ts_insights import \
    _print_ts_insights
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved
from offline.evolvers.skill_evolver_single_params import SkillEvolverParams
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, evolve_single_skill


def run_step(skills_root, skill_name, model, iterations, output_gepa_gated, ts_state_dir,
         verbose: bool = False, baseline_score=None, run_index: int = 1, n_runs: int = 1,
         scoring_mode: str = "single", baseline_score_multi=None, baseline_dims_multi=None,
         prebuilt_skill=None, prebuilt_dataset=None, prebuilt_baseline_module=None,
         prebuilt_trainset=None, prebuilt_valset=None, console=None):
    _banner("④ GEPA — TS-AcceptanceGate only (all examples equal weight)", run_index=run_index,
            n_runs=n_runs, console=console)
    console.print("  TS-TrainingSelector : OFF (all training examples, equal weight)")
    console.print()
    console.print(f"  TS-AcceptanceGate   : P(candidate > deployed) ≥ 0.75")
    console.print("    Monte Carlo (100 draws) — prevents accepting a lucky one-off run")
    console.print()

    evolver_config = EvolverConfig(
        skills_root = skills_root,
        output_dir = output_gepa_gated,
        iterations = iterations,
        optimizer_model = model,
        eval_model = model,
        max_prompt_growth=20.0,  # allow up to 2000% growth; GEPA now rewrites the full skill body
        verbose=verbose,
        # TS-TrainingSelector OFF, TS-AcceptanceGate ON
        ts_skill_scheduler = False,
        ts_example_selector = False,
        ts_acceptance_gate = True,
        ts_acceptance_confidence = 0.75,
        ts_acceptance_n_samples = 100,
        ts_state_dir = ts_state_dir,
        scoring_mode=scoring_mode,
    )

    params: SkillEvolverParams = SkillEvolverParams(skill_name,
                                                    "golden",
                                                    config=evolver_config,
                                                    min_improvement=0.0,
                                                    prior_baseline_score_single=baseline_score,
                                                    prior_baseline_score_multi=baseline_score_multi,
                                                    prior_baseline_dims_multi=baseline_dims_multi,
                                                    prebuilt_skill=prebuilt_skill,
                                                    prebuilt_dataset=prebuilt_dataset,
                                                    prebuilt_baseline_module=prebuilt_baseline_module,
                                                    prebuilt_trainset=prebuilt_trainset,
                                                    prebuilt_valset=prebuilt_valset,
                                                    console=console)
    metrics_gepa_gated = evolve_single_skill(params)
    _print_ts_insights(ts_state_dir, skill_name, console)

    if verbose:
        evolved_l3 = _read_latest_evolved(output_gepa_gated, skill_name)
        _print_skill("  Evolved skill (TS-AcceptanceGate only)", evolved_l3 or "[not produced]", console)

    return metrics_gepa_gated
