
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_config import DemoConfig
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_params import DemoParams
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_02_run_gepa_without_ts import \
    step as step_02_run_gepa_without_ts
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_03_run_gepa_with_ts_l2_l3 import \
    step as step_03_run_gepa_with_ts_l2_l3
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_04_run_gepa_with_ts_l2_only import \
    step as step_04_run_gepa_with_ts_l2_only
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_05_run_gepa_with_ts_l3_only import \
    step as step_05_run_gepa_with_ts_l3_only
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_restore_baseline_skill import \
    step as step_restore_baseline_skill


class DemoTrainingsResults:
    def __init__(self, runs, metrics_no_ts, metrics_l2_l3, metrics_l2, metrics_l3):
        self.runs = runs
        self.metrics_no_ts = metrics_no_ts
        self.metrics_l2_l3 = metrics_l2_l3
        self.metrics_l2 = metrics_l2
        self.metrics_l3 = metrics_l3

