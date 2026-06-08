from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.steps_shared_object import \
    SharedEvolutionObjects
from openjiuwen.agent_evolving_hermes.offline.evolvers._console_maker import _make_console
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_config import DemoConfig
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_params import DemoParams
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.trainings.trainings import DemoTrainings
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.trainings.results import (
    DemoTrainingsResults,
    run_key_label,
)
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
        3. For each (mode, fitness_metric) combination in order:
           a. Restore the baseline skill.
           b. Run the corresponding GEPA pass.
           c. Record the evolved metrics.
        4. Print the side-by-side comparison table (if any modes ran).
        5. Print output-file locations.

        Controlling which passes run
        ----------------------------
        Set ``run_modes`` and ``fitness_metrics`` in ``config.json``.

        Valid ``run_modes``:
        * ``"gepa_uniform"``              — plain GEPA, all training examples, threshold gate
        * ``"gepa_focused_on_difficulty"`` — TS Example Selector; focuses on discriminating examples
        * ``"gepa_gated"``                — TS Acceptance Gate; requires P(better) ≥ 0.75
        * ``"gepa_full"``                 — both TS levels active simultaneously
        * ``"gepa_rubric"``               — 5-dimension multi-objective scoring

        Valid ``fitness_metrics`` (used inside the GEPA optimizer loop):
        * ``"jiuwen"``  — stop-word-filtered weighted F1 (general-purpose, default)
        * ``"hermes"``  — word-bag with 0.3 floor (matches original Hermes)
        * Custom: dotted import path to any ``(example, prediction) -> float`` callable

        Use ``run_modes: []`` to run only the baseline holdout evaluation (no GEPA training).
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
        shared_evolution_object: SharedEvolutionObjects = (
            step_01_build_skill_dataset_and_dspy(params.skills_root,
                                                 params.skill_name,
                                                 self._config.model,
                                                 params.output_baseline,
                                                 console=console,
                                                 verbose=self._config.verbose))

        # ── Step 02: Evaluate baseline on holdout (NO training) ───────────────
        baseline_score_single, baseline_score_multi, multi_baseline_dims = (
            step_02_evaluate_baseline(shared_evolution_object=shared_evolution_object,
                                      skills_root=params.skills_root,
                                      model=self._config.model,
                                      output_dir=params.output_baseline,
                                      run_modes=self._config.run_modes,
                                      console=console,
                                      verbose=self._config.verbose))

        # ── Training passes (Steps 03, 04, 05, 06) ──────────────────────────────
        trainings_results: DemoTrainingsResults = self._trainings.run(
            params,
            baseline_score_single=baseline_score_single,
            baseline_score_multi=baseline_score_multi,
            baseline_dims_multi=multi_baseline_dims,
            shared_evolution_object=shared_evolution_object,
            console=console,
        )

        # ── Step 07: Comparison table (skip when ≤ 1 mode ran) ──────────────────
        if len(trainings_results.runs) >= 2:
            step_07_results_comparison(
                baseline_score_single,
                baseline_score_multi,
                scores=trainings_results.scores,
                metrics=trainings_results.metrics,
                ts_batch_size=self._config.ts_batch_size,
                console=console,
            )

        # ── Optional: Skill diff (baseline vs winner) ─────────────────────────
        if self._config.print_skill_diff and trainings_results.runs:
            self._print_skill_diff(params, trainings_results, console)

        # ── Step 08: Plots ─────────────────────────────────────────────────────
        step_08_plot_results(
            baseline_score_single,
            baseline_score_multi,
            scores=trainings_results.scores,
            output_dir=params.workdir / "plots",
            scenario_name=params.skill_name,
            n_runs=self._config.n_runs,
            console=console,
        )

        # ── Step 09: Where to look ─────────────────────────────────────────────
        step_09_final_prints(params.skill_name, trainings_results.runs, params.ts_state_dir, console)

    @staticmethod
    def _print_skill_diff(params: DemoParams, results: DemoTrainingsResults, console) -> None:
        """Determine winner and print baseline vs winner skill side by side."""
        from statistics import mean as _mean

        label_to_dir = dict(results.runs)
        mode_entries = [
            (run_key_label(k), results.scores[k], results.metrics.get(k))
            for k in results.scores
            if results.scores[k]
        ]
        present = [(l, s, m) for l, s, m in mode_entries if l in label_to_dir]
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
