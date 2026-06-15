
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.steps_shared_object import \
    SharedEvolutionObjects
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_single_params import SkillEvolverParams
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, evolve_single_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved


def run_step(shared_evolution_object: SharedEvolutionObjects,
             skills_root,
             skill_name,
             model,
             iterations,
             output_dir,
             console,
             verbose: bool = False,
             baseline_score_holistic=None,
             run_index: int = 1,
             n_runs: int = 1,
             scoring_mode: str = "holistic",
             baseline_score_rubrics=None,
             baseline_dims_rubrics=None,
             baseline_score_graph=None,
             baseline_score_checklist=None,
             baseline_score_instruction_following=None,
             baseline_score_consistency=None,
             fitness_metric: str = "bag_of_words",
             fitness_metric_fn_override=None):
    console.print(f"\n[bold cyan]*** Demo Step 03: Run GEPA Plain Started ***[/bold cyan]")

    _banner(f"② GEPA - Plain", run_index=run_index, n_runs=n_runs, console=console)
    console.print("  Example selector : all training examples, equal weight")
    console.print("  Acceptance gate  : threshold only (improvement ≥ 0.0)")
    console.print(f"  Fitness metric   : {fitness_metric}")
    console.print(f"  Holdout scoring  : {scoring_mode}")
    console.print()

    evolver_config = EvolverConfig(skills_root=skills_root,
                                   output_dir=output_dir,
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
                                   fitness_metric=fitness_metric,
                                   fitness_metric_fn_override=fitness_metric_fn_override)

    params: SkillEvolverParams = SkillEvolverParams(skill_name,
                                                    "golden",
                                                    config=evolver_config,
                                                    min_improvement=0.0,
                                                    prior_baseline_score_holistic=baseline_score_holistic,
                                                    prior_baseline_score_rubrics=baseline_score_rubrics,
                                                    prior_baseline_score_graph=baseline_score_graph,
                                                    prior_baseline_score_checklist=baseline_score_checklist,
                                                    prior_baseline_score_instruction_following=baseline_score_instruction_following,
                                                    prior_baseline_score_consistency=baseline_score_consistency,
                                                    prior_baseline_dims_rubrics=baseline_dims_rubrics,
                                                    prebuilt_skill=shared_evolution_object.skill,
                                                    prebuilt_dataset=shared_evolution_object.dataset,
                                                    prebuilt_baseline_module=shared_evolution_object.baseline_module,
                                                    prebuilt_trainset=shared_evolution_object.trainset,
                                                    prebuilt_valset=shared_evolution_object.valset,
                                                    console=console)
    metrics_gepa_plain = evolve_single_skill(params)

    if verbose:
        evolved_gepa_plain = _read_latest_evolved(output_dir, skill_name)
        _print_skill("  Evolved skill (no TS)", evolved_gepa_plain or "[not produced]", console)

    console.print(f"[bold cyan]*** Demo Step 03: Run GEPA Plain Finished ***[/bold cyan]")
    return metrics_gepa_plain
