
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_config import DemoConfig
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_params import DemoParams
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_00_evaluate_holdout import \
    step as step_00_evaluate_holdout
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_01_save_skill_and_dataset import \
    step as step_01_save_skill_and_dataset
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_02_run_gepa_without_thompson_sampling import \
    step as step_02_run_gepa_without_thompson_sampling
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_03_restore_baseline_skill import \
    step as step_03_restore_baseline_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_04b_run_gepa_l2_only import \
    step as step_04b_run_gepa_l2_only
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_04c_run_gepa_l3_only import \
    step as step_04c_run_gepa_l3_only
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_04_run_gepa_with_thompson_sampling import \
    step as step_04_run_gepa_with_thompson_sampling
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_05_results_comparison import \
    step as step_05_comparison
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_06_final_prints import \
    step as step_06_final_prints


class Demo:
    def __init__(self, config: DemoConfig):
        self.config = config

    def run(self, params: DemoParams) -> None:
        """Run the full Thompson Sampling vs baseline demonstration.

        Flow
        ----
        1. Save baseline skill + golden dataset to disk.
        2. Evaluate the baseline skill on holdout (no training).
        3. For each mode in ``params.run_modes`` (in order):
           a. Restore the baseline skill.
           b. Run the corresponding GEPA pass.
           c. Record the evolved metrics.
        4. Print the side-by-side comparison table (if any modes ran).
        5. Print output-file locations.

        Controlling which passes run
        ----------------------------
        ``params.run_modes`` is a list of mode strings.  Valid values:

        * ``"no_ts"``   — plain GEPA, all training examples, threshold gate
        * ``"l2_only"`` — TS Example Selector; focuses on discriminating examples
        * ``"l3_only"`` — TS Acceptance Gate; requires P(better) ≥ 0.75
        * ``"l2_l3"``   — both TS levels active simultaneously

        Pass ``[]`` to run only the baseline holdout evaluation (no GEPA training).
        Pass ``["no_ts"]`` to run just the No-TS pass, etc.
        """

        # ── Step 1: Write baseline skill + golden dataset ─────────────────────
        step_01_save_skill_and_dataset(
            params.skills_root, params.skill_name,
            params.skill_body, params.skill_frontmatter, params.golden_examples,
            verbose=params.verbose,
        )

        # ── Step 0: Evaluate baseline on holdout (NO training) ───────────────
        baseline_score: float = step_00_evaluate_holdout(
            params.skills_root, params.skill_name, params.model,
            params.output_baseline, params.verbose,
        )

        # ── Training passes ───────────────────────────────────────────────────
        metrics_no_ts  = None
        metrics_l2     = None
        metrics_l3     = None
        metrics_l2_l3  = None
        runs: list[tuple[str, object]] = []

        def _restore() -> None:
            step_03_restore_baseline_skill(
                params.skills_root, params.skill_name,
                params.skill_frontmatter, params.skill_body,
            )

        if "no_ts" in params.run_modes:
            _restore()
            metrics_no_ts = step_02_run_gepa_without_thompson_sampling(
                params.skills_root, params.skill_name, params.model,
                params.iterations, params.output_no_ts, verbose=params.verbose,
            )
            runs.append(("No-TS", params.output_no_ts))

        if "l2_only" in params.run_modes:
            _restore()
            metrics_l2 = step_04b_run_gepa_l2_only(
                params.skills_root, params.skill_name, params.model,
                params.iterations, params.ts_batch_size,
                params.output_l2_only, params.ts_state_dir, verbose=params.verbose,
            )
            runs.append(("L2-only", params.output_l2_only))

        if "l3_only" in params.run_modes:
            _restore()
            metrics_l3 = step_04c_run_gepa_l3_only(
                params.skills_root, params.skill_name, params.model,
                params.iterations, params.output_l3_only, params.ts_state_dir,
                verbose=params.verbose,
            )
            runs.append(("L3-only", params.output_l3_only))

        if "l2_l3" in params.run_modes:
            _restore()
            metrics_l2_l3 = step_04_run_gepa_with_thompson_sampling(
                params.skills_root, params.skill_name, params.model,
                params.iterations, params.ts_batch_size, params.golden_examples,
                params.output_l2_l3, params.ts_state_dir, verbose=params.verbose,
            )
            runs.append(("L2+L3", params.output_l2_l3))

        # ── Step 5: Comparison table (skip when ≤ 1 mode ran — no value in repeating) ──
        if len(runs) >= 2:
            step_05_comparison(
                baseline_score,
                metrics_no_ts=metrics_no_ts,
                metrics_l2=metrics_l2,
                metrics_l3=metrics_l3,
                metrics_l2_l3=metrics_l2_l3,
                ts_batch_size=params.ts_batch_size,
            )

        # ── Step 6: Where to look ─────────────────────────────────────────────
        step_06_final_prints(params.skill_name, runs, params.ts_state_dir)
