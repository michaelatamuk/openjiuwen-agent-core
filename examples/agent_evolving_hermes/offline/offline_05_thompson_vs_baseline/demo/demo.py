from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_config import DemoConfig
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_params import DemoParams
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_trainings import DemoTrainings
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_trainings_results import \
    DemoTrainingsResults
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_00_save_skill_and_dataset import \
    step as step_00_save_skill_and_dataset
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_01_evaluate_baseline import \
    step as step_01_evaluate_baseline
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_06_results_comparison import \
    step as step_06_results_comparison
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_07_plot_results import \
    step as step_07_plot_results
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_08_final_prints import \
    step as step_08_final_prints


class Demo:
    def __init__(self, config: DemoConfig):
        self._config = config
        self._trainings: DemoTrainings = DemoTrainings(config)

    def run(self, params: DemoParams) -> None:
        """Run the full Thompson Sampling vs baseline demonstration.

        Flow
        ----
        1. Save baseline skill + golden dataset to disk.
        2. Evaluate the baseline skill on holdout (no training).
        3. For each mode in ``self.config.run_modes`` (in order):
           a. Restore the baseline skill.
           b. Run the corresponding GEPA pass.
           c. Record the evolved metrics.
        4. Print the side-by-side comparison table (if any modes ran).
        5. Print output-file locations.

        Controlling which passes run
        ----------------------------
        Set ``run_modes`` in ``config.json``.  Valid values:

        * ``"gepa_uniform"``   — plain GEPA, all training examples, threshold gate
        * ``"gepa_focused_on_difficulty"`` — TS Example Selector; focuses on discriminating examples
        * ``"gepa_gated"`` — TS Acceptance Gate; requires P(better) ≥ 0.75
        * ``"gepa_full"``   — both TS levels active simultaneously

        Use ``[]`` to run only the baseline holdout evaluation (no GEPA training).
        """

        # ── Step 0: Write baseline skill + golden dataset ─────────────────────
        step_00_save_skill_and_dataset(params.skills_root, params.skill_name, params.skill_body,
                                       params.skill_frontmatter, params.golden_examples, verbose=self._config.verbose)

        # ── Step 1: Evaluate baseline on holdout (NO training) ───────────────
        # Evaluates the single-score baseline unconditionally; also evaluates the
        # multi-objective baseline when "gepa_rubric" is in run_modes so that
        # GEPA runs never need to re-evaluate the baseline themselves.
        baseline_score_single, baseline_score_multi, multi_baseline_dims = step_01_evaluate_baseline(
            params.skills_root, params.skill_name, self._config.model,
            params.output_baseline, self._config.verbose,
            run_modes=self._config.run_modes,
        )

        # ── Training passes ───────────────────────────────────────────────────
        trainings_results: DemoTrainingsResults = self._trainings.run(
            params, baseline_score=baseline_score_single, multi_baseline_dims=multi_baseline_dims,
        )

        # ── Step 6: Comparison table (skip when ≤ 1 mode ran) ────────────────
        if len(trainings_results.runs) >= 2:
            step_06_results_comparison(baseline_score_single,
                                       baseline_score_multi,
                                       scores_gepa_uniform=trainings_results.scores_gepa_uniform or None,
                                       scores_gepa_full=trainings_results.scores_gepa_full or None,
                                       scores_gepa_focused=trainings_results.scores_gepa_focused or None,
                                       scores_gepa_gated=trainings_results.scores_gepa_gated or None,
                                       scores_gepa_rubric=trainings_results.scores_gepa_rubric or None,
                                       metrics_gepa_uniform=trainings_results.metrics_gepa_uniform,
                                       metrics_gepa_full=trainings_results.metrics_gepa_full,
                                       metrics_gepa_focused=trainings_results.metrics_gepa_focused,
                                       metrics_gepa_gated=trainings_results.metrics_gepa_gated,
                                       metrics_gepa_rubric=trainings_results.metrics_gepa_rubric,
                                       ts_batch_size=self._config.ts_batch_size)

        # ── Step 8: Plots ─────────────────────────────────────────────────────
        step_07_plot_results(baseline_score_single,
                             baseline_score_multi,
                             scores_gepa_uniform=trainings_results.scores_gepa_uniform or None,
                             scores_gepa_rubric=trainings_results.scores_gepa_rubric or None,
                             scores_gepa_full=trainings_results.scores_gepa_full or None,
                             scores_gepa_focused=trainings_results.scores_gepa_focused or None,
                             scores_gepa_gated=trainings_results.scores_gepa_gated or None,
                             output_dir=params.workdir / "plots",
                             scenario_name=params.skill_name,
                             n_runs=self._config.n_runs)

        # ── Step 7: Where to look ─────────────────────────────────────────────
        step_08_final_prints(params.skill_name, trainings_results.runs, params.ts_state_dir)
