
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_config import DemoConfig
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_params import DemoParams
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_trainings_results import \
    DemoTrainingsResults
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_02_run_gepa_without_ts import \
    step as step_02_run_gepa_without_ts
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_03_run_gepa_with_ts_l2_l3 import \
    step as step_03_run_gepa_with_ts_l2_l3
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_04_run_gepa_with_ts_l2_only import \
    step as step_04_run_gepa_with_ts_l2_only
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_05_run_gepa_with_ts_l3_only import \
    step as step_05_run_gepa_with_ts_l3_only
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_restore_baseline_skill import \
    step as step_restore_baseline_skill


class DemoTrainings:
    def __init__(self, config: DemoConfig):
        self._config = config

    def run(self, params: DemoParams):
        runs: list[tuple[str, object]] = []

        metrics_no_ts = None
        if "no_ts" in self._config.run_modes:
            self._restore_baseline_skill(params)
            metrics_no_ts = step_02_run_gepa_without_ts(
                params.skills_root, params.skill_name, self._config.model,
                self._config.iterations, params.output_no_ts, verbose=self._config.verbose,
            )
            runs.append(("No-TS", params.output_no_ts))

        metrics_l2_l3 = None
        if "l2_l3" in self._config.run_modes:
            self._restore_baseline_skill(params)
            metrics_l2_l3 = step_03_run_gepa_with_ts_l2_l3(
                params.skills_root, params.skill_name, self._config.model,
                self._config.iterations, self._config.ts_batch_size, params.golden_examples,
                params.output_l2_l3, params.ts_state_dir, verbose=self._config.verbose,
            )
            runs.append(("L2+L3", params.output_l2_l3))

        metrics_l2 = None
        if "l2_only" in self._config.run_modes:
            self._restore_baseline_skill(params)
            metrics_l2 = step_04_run_gepa_with_ts_l2_only(
                params.skills_root, params.skill_name, self._config.model,
                self._config.iterations, self._config.ts_batch_size,
                params.output_l2_only, params.ts_state_dir, verbose=self._config.verbose,
            )
            runs.append(("L2-only", params.output_l2_only))

        metrics_l3 = None
        if "l3_only" in self._config.run_modes:
            self._restore_baseline_skill(params)
            metrics_l3 = step_05_run_gepa_with_ts_l3_only(
                params.skills_root, params.skill_name, self._config.model,
                self._config.iterations, params.output_l3_only, params.ts_state_dir,
                verbose=self._config.verbose,
            )
            runs.append(("L3-only", params.output_l3_only))

        trainings_results: DemoTrainingsResults = DemoTrainingsResults(runs, metrics_no_ts, metrics_l2_l3, metrics_l2,
                                                                       metrics_l3)
        return trainings_results

    @staticmethod
    def _restore_baseline_skill(params) -> None:
        step_restore_baseline_skill(
            params.skills_root, params.skill_name,
            params.skill_frontmatter, params.skill_body,
        )
