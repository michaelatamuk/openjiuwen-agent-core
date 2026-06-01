
from __future__ import annotations

from pathlib import Path

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

    def run(self, params: DemoParams) -> DemoTrainingsResults:
        n = self._config.n_runs
        cfg = self._config

        # Helper: return run-specific output and TS-state dirs.
        # When n==1 use the canonical path so existing single-run output
        # structure is unchanged; for n>1 suffix with _r{i}.
        def _out(base: str, i: int) -> Path:
            return params.workdir / base if n == 1 else params.workdir / f"{base}_r{i}"

        def _ts(i: int) -> Path:
            return params.ts_state_dir if n == 1 else params.workdir / f"ts_state_r{i}"

        # ── no_ts ─────────────────────────────────────────────────────────────
        scores_no_ts: list[float] = []
        metrics_no_ts = None
        last_out_no_ts = params.output_no_ts
        if "no_ts" in cfg.run_modes:
            for i in range(1, n + 1):
                out = _out("output_no_ts", i)
                self._restore(params)
                m = step_02_run_gepa_without_ts(
                    params.skills_root, params.skill_name, cfg.model,
                    cfg.iterations, out, verbose=cfg.verbose,
                )
                scores_no_ts.append(m.get("evolved_score", 0.0))
                metrics_no_ts = m
                last_out_no_ts = out

        # ── l2_l3 ─────────────────────────────────────────────────────────────
        scores_l2_l3: list[float] = []
        metrics_l2_l3 = None
        last_out_l2_l3 = params.output_l2_l3
        if "l2_l3" in cfg.run_modes:
            for i in range(1, n + 1):
                out = _out("output_l2_l3", i)
                ts  = _ts(i)
                self._restore(params)
                m = step_03_run_gepa_with_ts_l2_l3(
                    params.skills_root, params.skill_name, cfg.model,
                    cfg.iterations, cfg.ts_batch_size, params.golden_examples,
                    out, ts, verbose=cfg.verbose,
                )
                scores_l2_l3.append(m.get("evolved_score", 0.0))
                metrics_l2_l3 = m
                last_out_l2_l3 = out

        # ── l2_only ───────────────────────────────────────────────────────────
        scores_l2: list[float] = []
        metrics_l2 = None
        last_out_l2 = params.output_l2_only
        if "l2_only" in cfg.run_modes:
            for i in range(1, n + 1):
                out = _out("output_l2_only", i)
                ts  = _ts(i)
                self._restore(params)
                m = step_04_run_gepa_with_ts_l2_only(
                    params.skills_root, params.skill_name, cfg.model,
                    cfg.iterations, cfg.ts_batch_size,
                    out, ts, verbose=cfg.verbose,
                )
                scores_l2.append(m.get("evolved_score", 0.0))
                metrics_l2 = m
                last_out_l2 = out

        # ── l3_only ───────────────────────────────────────────────────────────
        scores_l3: list[float] = []
        metrics_l3 = None
        last_out_l3 = params.output_l3_only
        if "l3_only" in cfg.run_modes:
            for i in range(1, n + 1):
                out = _out("output_l3_only", i)
                ts  = _ts(i)
                self._restore(params)
                m = step_05_run_gepa_with_ts_l3_only(
                    params.skills_root, params.skill_name, cfg.model,
                    cfg.iterations, out, ts, verbose=cfg.verbose,
                )
                scores_l3.append(m.get("evolved_score", 0.0))
                metrics_l3 = m
                last_out_l3 = out

        # ── build runs list (label → last run's output dir) ───────────────────
        runs: list[tuple[str, object]] = []
        if scores_no_ts:  runs.append(("No-TS",   last_out_no_ts))
        if scores_l2_l3:  runs.append(("L2+L3",   last_out_l2_l3))
        if scores_l2:     runs.append(("L2-only",  last_out_l2))
        if scores_l3:     runs.append(("L3-only",  last_out_l3))

        return DemoTrainingsResults(
            runs=runs,
            scores_no_ts=scores_no_ts,
            scores_l2_l3=scores_l2_l3,
            scores_l2=scores_l2,
            scores_l3=scores_l3,
            metrics_no_ts=metrics_no_ts,
            metrics_l2_l3=metrics_l2_l3,
            metrics_l2=metrics_l2,
            metrics_l3=metrics_l3,
        )

    @staticmethod
    def _restore(params: DemoParams) -> None:
        step_restore_baseline_skill(
            params.skills_root, params.skill_name,
            params.skill_frontmatter, params.skill_body,
        )
