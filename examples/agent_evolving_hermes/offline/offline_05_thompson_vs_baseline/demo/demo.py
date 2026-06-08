from __future__ import annotations

from openjiuwen.agent_evolving_hermes.offline.evolvers._console_maker import _make_console
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_config import DemoConfig
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_params import DemoParams
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.trainings.trainings import DemoTrainings
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.trainings.results import \
    DemoTrainingsResults
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_00_write_demo_scenario_files import \
    run_step as step_00_write_demo_scenario_files
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_01_build_skill_dataset_and_dspy import \
    run_step as step_01_build_skill_dataset_and_dspy
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_02_evaluate_baseline import \
    run_step as step_02_evaluate_baseline
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_07_results_comparison import \
    run_step as step_07_results_comparison
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_08_plot_results import \
    run_step as step_08_plot_results
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_09_final_prints import \
    run_step as step_09_final_prints
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill_diff import \
    print_skill_diff
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.reader_latest_evolved import \
    _read_latest_evolved


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

        console = _make_console()

        # ── Step 00: Write demo scenario files to disk (demo-only; not needed in production) ──
        step_00_write_demo_scenario_files(params.skills_root,
                                          params.skill_name,
                                          params.skill_body,
                                          params.skill_frontmatter,
                                          params.golden_examples,
                                          verbose=self._config.verbose,
                                          console=console)

        # ── Step 01: Build skill / dataset / DSPy ONCE (stages 1–4) ────────────
        # Runs find_and_load_skill / validate_baseline_constraints /
        # build_or_load_dataset / configure_dspy_and_prepare_sets exactly once.
        # The resulting objects are passed to both step_01 and all GEPA
        # training passes so these stages never execute more than once per run.
        shared_evolution_object: SharedEvolutionObjects = (
            step_01_build_skill_dataset_and_dspy(params.skills_root,
                                                 params.skill_name,
                                                 self._config.model,
                                                 params.output_baseline,
                                                 verbose=self._config.verbose,
                                                 console=console))

        # ── Step 02: Evaluate baseline on holdout (NO training) ───────────────
        # Evaluates the single-score baseline unconditionally; also evaluates the
        # multi-objective baseline when "gepa_rubric" is in run_modes so that
        # GEPA runs never need to re-evaluate the baseline themselves.
        # Prebuilt objects are passed so stages 1 / 3 / 4 are skipped here.
        baseline_score_single, baseline_score_multi, multi_baseline_dims = (
            step_02_evaluate_baseline(params.skills_root,
                                      self._config.model,
                                      params.output_baseline,
                                      self._config.verbose,
                                      run_modes=self._config.run_modes,
                                      shared_evolution_object=shared_evolution_object,
                                      console=console))

        # ── Training passes (Steps 03, 04, 05, 06) ───────────────────────────────────────────────────
        trainings_results: DemoTrainingsResults = self._trainings.run(params,
                                                                      baseline_score_single=baseline_score_single,
                                                                      baseline_score_multi=baseline_score_multi,
                                                                      baseline_dims_multi=multi_baseline_dims,
                                                                      shared_evolution_object=shared_evolution_object,
                                                                      console=console)

        # ── Step 07: Comparison table (skip when ≤ 1 mode ran) ────────────────
        if len(trainings_results.runs) >= 2:
            step_07_results_comparison(baseline_score_single,
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
                                       ts_batch_size=self._config.ts_batch_size,
                                       console=console)

        # ── Optional: Skill diff (baseline vs winner) ─────────────────────────
        if self._config.print_skill_diff and trainings_results.runs:
            self._print_skill_diff(params, trainings_results, console)

        # ── Step 08: Plots ─────────────────────────────────────────────────────
        step_08_plot_results(baseline_score_single,
                             baseline_score_multi,
                             scores_gepa_uniform=trainings_results.scores_gepa_uniform or None,
                             scores_gepa_rubric=trainings_results.scores_gepa_rubric or None,
                             scores_gepa_full=trainings_results.scores_gepa_full or None,
                             scores_gepa_focused=trainings_results.scores_gepa_focused or None,
                             scores_gepa_gated=trainings_results.scores_gepa_gated or None,
                             output_dir=params.workdir / "plots",
                             scenario_name=params.skill_name,
                             n_runs=self._config.n_runs,
                             console=console)

        # ── Step 09: Where to look ─────────────────────────────────────────────
        step_09_final_prints(params.skill_name, trainings_results.runs, params.ts_state_dir, console)

    @staticmethod
    def _print_skill_diff(params: DemoParams, results: DemoTrainingsResults, console) -> None:
        """Determine winner and print baseline vs winner skill side by side."""
        from statistics import mean as _mean

        label_to_dir = dict(results.runs)
        mode_entries = [
            ("GEPA-Uniform",  results.scores_gepa_uniform,  results.metrics_gepa_uniform),
            ("GEPA-Rubric",   results.scores_gepa_rubric,   results.metrics_gepa_rubric),
            ("GEPA-Focused",  results.scores_gepa_focused,  results.metrics_gepa_focused),
            ("GEPA-Gated",    results.scores_gepa_gated,    results.metrics_gepa_gated),
            ("GEPA-Full",     results.scores_gepa_full,     results.metrics_gepa_full),
        ]
        present = [(l, s, m) for l, s, m in mode_entries if s and l in label_to_dir]
        if not present:
            return

        accepted = [(l, _mean(s), m) for l, s, m in present if m and m.get("accepted")]
        pool = accepted if accepted else [(l, _mean(s), m) for l, s, m in present]
        best_score = max(sc for _, sc, _ in pool)
        winner_label = next(l for l, sc, _ in pool if sc == best_score)

        baseline_path = params.skills_root / params.skill_name / "SKILL.md"
        if not baseline_path.exists():
            return
        winner_text = _read_latest_evolved(label_to_dir[winner_label], params.skill_name)
        if not winner_text:
            return

        print_skill_diff(baseline_path.read_text(),
                         winner_label,
                         winner_text,
                         winner_score=best_score,
                         console=console)
