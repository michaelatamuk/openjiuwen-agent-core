
from __future__ import annotations

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_single_params import SkillEvolverParams
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, evolve_single_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved


def run_step(skills_root, skill_name, model, iterations, output_gepa_uniform, verbose: bool = False,
         baseline_score_single=None, run_index: int = 1, n_runs: int = 1,
         scoring_mode: str = "single", baseline_score_multi=None, baseline_dims_multi=None,
         prebuilt_skill=None, prebuilt_dataset=None, prebuilt_baseline_module=None,
         prebuilt_trainset=None, prebuilt_valset=None, console=None):
    console.print(f"\n[bold cyan]*** Demo Step 03: Run GEPA Uniform Started ***[/bold cyan]")

    _banner(f"② GEPA — without Thompson Sampling (scoring mode {scoring_mode})", run_index=run_index,
            n_runs=n_runs, console=console)
    console.print("  Example selector : all training examples, equal weight")
    console.print("  Acceptance gate  : threshold only (improvement ≥ 0.0)")
    console.print()

    evolver_config = EvolverConfig(
        skills_root=skills_root,
        output_dir=output_gepa_uniform,
        iterations=iterations,
        optimizer_model=model,
        eval_model=model,
        max_prompt_growth=20.0,  # allow up to 2000% growth; GEPA now rewrites the full skill body
        verbose=verbose,
        # Thompson Sampling — all OFF
        ts_skill_scheduler=False,
        ts_example_selector=False,
        ts_acceptance_gate=False,
        scoring_mode=scoring_mode,
    )

    params: SkillEvolverParams = SkillEvolverParams(skill_name,
                                                    "golden",
                                                    config=evolver_config,
                                                    min_improvement=0.0,
                                                    prior_baseline_score_single=baseline_score_single,
                                                    prior_baseline_score_multi=baseline_score_multi,
                                                    prior_baseline_dims_multi=baseline_dims_multi,
                                                    prebuilt_skill=prebuilt_skill,
                                                    prebuilt_dataset=prebuilt_dataset,
                                                    prebuilt_baseline_module=prebuilt_baseline_module,
                                                    prebuilt_trainset=prebuilt_trainset,
                                                    prebuilt_valset=prebuilt_valset,
                                                    console=console)
    metrics_gepa_uniform = evolve_single_skill(params)

    if verbose:
        evolved_gepa_uniform = _read_latest_evolved(output_gepa_uniform, skill_name)
        _print_skill("  Evolved skill (no TS)", evolved_gepa_uniform or "[not produced]", console)

    console.print(f"[bold cyan]*** Demo Step 03: Run GEPA Uniform Finished ***[/bold cyan]")
    return metrics_gepa_uniform
