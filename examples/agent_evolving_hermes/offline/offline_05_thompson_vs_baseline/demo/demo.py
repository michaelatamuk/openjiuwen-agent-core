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
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_07_final_prints import \
    step as step_07_final_prints


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

        * ``"no_ts"``   — plain GEPA, all training examples, threshold gate
        * ``"l2_only"`` — TS Example Selector; focuses on discriminating examples
        * ``"l3_only"`` — TS Acceptance Gate; requires P(better) ≥ 0.75
        * ``"l2_l3"``   — both TS levels active simultaneously

        Use ``[]`` to run only the baseline holdout evaluation (no GEPA training).
        """

        # ── Step 0: Write baseline skill + golden dataset ─────────────────────
        step_00_save_skill_and_dataset(
            params.skills_root, params.skill_name,
            params.skill_body, params.skill_frontmatter, params.golden_examples,
            verbose=self._config.verbose,
        )

        # ── Step 0: Evaluate baseline on holdout (NO training) ───────────────
        baseline_score: float = step_01_evaluate_baseline(
            params.skills_root, params.skill_name, self._config.model,
            params.output_baseline, self._config.verbose,
        )

        # ── Training passes ───────────────────────────────────────────────────
        trainings_results: DemoTrainingsResults = self._trainings.run(params)

        # ── Step 6: Comparison table (skip when ≤ 1 mode ran) ────────────────
        if len(trainings_results.runs) >= 2:
            step_06_results_comparison(
                baseline_score,
                scores_no_ts=trainings_results.scores_no_ts or None,
                scores_l2_l3=trainings_results.scores_l2_l3 or None,
                scores_l2=trainings_results.scores_l2 or None,
                scores_l3=trainings_results.scores_l3 or None,
                metrics_no_ts=trainings_results.metrics_no_ts,
                metrics_l2_l3=trainings_results.metrics_l2_l3,
                metrics_l2=trainings_results.metrics_l2,
                metrics_l3=trainings_results.metrics_l3,
                ts_batch_size=self._config.ts_batch_size,
            )

        # ── Step 7: Where to look ─────────────────────────────────────────────
        step_07_final_prints(params.skill_name, trainings_results.runs, params.ts_state_dir)
