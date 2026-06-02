
from __future__ import annotations

import time
from pathlib import Path

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_config import DemoConfig
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_params import DemoParams
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_trainings_results import \
    DemoTrainingsResults
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_mode_summary import \
    print_mode_summary, print_mode_timing
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.writer_skill import \
    _write_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_02_run_gepa_without_ts import \
    step as step_02_run_gepa_without_ts
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_03_run_gepa_with_ts_l2_only import \
    step as step_03_run_gepa_with_ts_l2_only
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_04_run_gepa_with_ts_l3_only import \
    step as step_04_run_gepa_with_ts_l3_only
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_05_run_gepa_with_ts_l2_l3 import \
    step as step_05_run_gepa_with_ts_l2_l3


class DemoTrainings:
    def __init__(self, config: DemoConfig):
        self._config: DemoConfig = config

    def run(
        self,
        params: DemoParams,
        baseline_score: float = None,
        multi_baseline_dims=None,
    ) -> DemoTrainingsResults:
        runs: list[tuple[str, object]] = []

        # ── no_ts ─────────────────────────────────────────────────────────────
        scores_no_ts, metrics_no_ts, last_out_no_ts = self.run_no_ts(params, baseline_score)
        if scores_no_ts:
            runs.append(("No-TS", last_out_no_ts))

        # ── no_ts_multi ───────────────────────────────────────────────────────
        scores_no_ts_multi, metrics_no_ts_multi, last_out_no_ts_multi = self.run_no_ts_multi(
            params, baseline_score, multi_baseline_dims=multi_baseline_dims,
        )
        if scores_no_ts_multi:
            runs.append(("No-TS-Multi", last_out_no_ts_multi))

        # ── l2_only ───────────────────────────────────────────────────────────
        scores_l2, metrics_l2, last_out_l2 = self.run_l2_only(params, baseline_score)
        if scores_l2:
            runs.append(("L2-only", last_out_l2))

        # ── l3_only ───────────────────────────────────────────────────────────
        scores_l3, metrics_l3, last_out_l3 = self.run_l3_only(params, baseline_score)
        if scores_l3:
            runs.append(("L3-only", last_out_l3))

        # ── l2_l3 ─────────────────────────────────────────────────────────────
        scores_l2_l3, metrics_l2_l3, last_out_l2_l3 = self.run_l2_l3(params, baseline_score)
        if scores_l2_l3:
            runs.append(("L2+L3", last_out_l2_l3))

        return DemoTrainingsResults(
            runs=runs,
            scores_no_ts=scores_no_ts,
            scores_l2_l3=scores_l2_l3,
            scores_l2=scores_l2,
            scores_l3=scores_l3,
            scores_no_ts_multi=scores_no_ts_multi,
            metrics_no_ts=metrics_no_ts,
            metrics_l2_l3=metrics_l2_l3,
            metrics_l2=metrics_l2,
            metrics_l3=metrics_l3,
            metrics_no_ts_multi=metrics_no_ts_multi,
        )

    def run_no_ts(self, params: DemoParams, baseline_score: float = None):
        scores_no_ts: list[float] = []
        metrics_no_ts = None
        last_out_no_ts = params.output_no_ts
        if "no_ts" in self._config.run_modes:
            t_start = time.monotonic()
            for i in range(1, self._config.n_runs + 1):
                out = self._out(params, "output_no_ts", i)
                self._step_restore_baseline_skill(params)
                m = step_02_run_gepa_without_ts(
                    params.skills_root, params.skill_name, self._config.model,
                    self._config.iterations, out, verbose=self._config.verbose,
                    baseline_score=baseline_score,
                    run_index=i, n_runs=self._config.n_runs,
                )
                scores_no_ts.append(m.get("evolved_score", 0.0))
                metrics_no_ts = m
                last_out_no_ts = out
            elapsed = time.monotonic() - t_start
            if len(scores_no_ts) > 1:
                print_mode_summary("No-TS", baseline_score, scores_no_ts, elapsed_sec=elapsed)
            else:
                print_mode_timing("No-TS", elapsed)
        return scores_no_ts, metrics_no_ts, last_out_no_ts

    def run_no_ts_multi(self, params: DemoParams, baseline_score: float = None, multi_baseline_dims=None):
        scores: list[float] = []
        metrics = None
        last_out = params.workdir / "output_no_ts_multi"
        if "no_ts_multi" in self._config.run_modes:
            t_start = time.monotonic()
            for i in range(1, self._config.n_runs + 1):
                out = self._out(params, "output_no_ts_multi", i)
                self._step_restore_baseline_skill(params)
                m = step_02_run_gepa_without_ts(
                    params.skills_root, params.skill_name, self._config.model,
                    self._config.iterations, out, verbose=self._config.verbose,
                    baseline_score=baseline_score,
                    run_index=i, n_runs=self._config.n_runs,
                    scoring_mode="multi",
                    multi_baseline_dims=multi_baseline_dims,
                )
                scores.append(m.get("evolved_score", 0.0))
                metrics = m
                last_out = out
            elapsed = time.monotonic() - t_start
            if len(scores) > 1:
                print_mode_summary("No-TS-Multi", baseline_score, scores, elapsed_sec=elapsed)
            else:
                print_mode_timing("No-TS-Multi", elapsed)
        return scores, metrics, last_out

    def run_l2_l3(self, params: DemoParams, baseline_score: float = None):
        scores_l2_l3: list[float] = []
        metrics_l2_l3 = None
        last_out_l2_l3 = params.output_l2_l3
        if "l2_l3" in self._config.run_modes:
            t_start = time.monotonic()
            for i in range(1, self._config.n_runs + 1):
                out = self._out(params, "output_l2_l3", i)
                ts  = self._ts(params, i)
                self._step_restore_baseline_skill(params)
                m = step_05_run_gepa_with_ts_l2_l3(
                    params.skills_root, params.skill_name, self._config.model,
                    self._config.iterations, self._config.ts_batch_size, params.golden_examples,
                    out, ts, verbose=self._config.verbose,
                    baseline_score=baseline_score,
                    run_index=i, n_runs=self._config.n_runs,
                )
                scores_l2_l3.append(m.get("evolved_score", 0.0))
                metrics_l2_l3 = m
                last_out_l2_l3 = out
            elapsed = time.monotonic() - t_start
            if len(scores_l2_l3) > 1:
                print_mode_summary("L2+L3", baseline_score, scores_l2_l3, elapsed_sec=elapsed)
            else:
                print_mode_timing("L2+L3", elapsed)
        return scores_l2_l3, metrics_l2_l3, last_out_l2_l3

    def run_l2_only(self, params: DemoParams, baseline_score: float = None):
        scores_l2: list[float] = []
        metrics_l2 = None
        last_out_l2 = params.output_l2_only
        if "l2_only" in self._config.run_modes:
            t_start = time.monotonic()
            for i in range(1, self._config.n_runs + 1):
                out = self._out(params, "output_l2_only", i)
                ts  = self._ts(params, i)
                self._step_restore_baseline_skill(params)
                m = step_03_run_gepa_with_ts_l2_only(
                    params.skills_root, params.skill_name, self._config.model,
                    self._config.iterations, self._config.ts_batch_size,
                    out, ts, verbose=self._config.verbose,
                    baseline_score=baseline_score,
                    run_index=i, n_runs=self._config.n_runs,
                )
                scores_l2.append(m.get("evolved_score", 0.0))
                metrics_l2 = m
                last_out_l2 = out
            elapsed = time.monotonic() - t_start
            if len(scores_l2) > 1:
                print_mode_summary("L2 only", baseline_score, scores_l2, elapsed_sec=elapsed)
            else:
                print_mode_timing("L2 only", elapsed)
        return scores_l2, metrics_l2, last_out_l2

    def run_l3_only(self, params: DemoParams, baseline_score: float = None):
        scores_l3: list[float] = []
        metrics_l3 = None
        last_out_l3 = params.output_l3_only
        if "l3_only" in self._config.run_modes:
            t_start = time.monotonic()
            for i in range(1, self._config.n_runs + 1):
                out = self._out(params, "output_l3_only", i)
                ts  = self._ts(params, i)
                self._step_restore_baseline_skill(params)
                m = step_04_run_gepa_with_ts_l3_only(
                    params.skills_root, params.skill_name, self._config.model,
                    self._config.iterations, out, ts, verbose=self._config.verbose,
                    baseline_score=baseline_score,
                    run_index=i, n_runs=self._config.n_runs,
                )
                scores_l3.append(m.get("evolved_score", 0.0))
                metrics_l3 = m
                last_out_l3 = out
            elapsed = time.monotonic() - t_start
            if len(scores_l3) > 1:
                print_mode_summary("L3 only", baseline_score, scores_l3, elapsed_sec=elapsed)
            else:
                print_mode_timing("L3 only", elapsed)
        return scores_l3, metrics_l3, last_out_l3

    @staticmethod
    def _step_restore_baseline_skill(params: DemoParams) -> None:
        _write_skill(params.skills_root, params.skill_name, params.skill_frontmatter, params.skill_body)

    # Helper: return run-specific output and TS-state dirs.
    # When n==1 use the canonical path so existing single-run output
    # structure is unchanged; for n>1 suffix with _r{i}.
    def _out(self, params: DemoParams, base: str, i: int) -> Path:
        return params.workdir / base if self._config.n_runs == 1 else params.workdir / f"{base}_r{i}"

    def _ts(self, params: DemoParams, i: int) -> Path:
        return params.ts_state_dir if self._config.n_runs == 1 else params.workdir / f"ts_state_r{i}"
