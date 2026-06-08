
from __future__ import annotations

import time
from pathlib import Path

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_config import DemoConfig
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_params import DemoParams
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.trainings.results import \
    DemoTrainingsResults
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_mode_summary import \
    print_mode_summary, print_mode_timing
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.writer_skill import \
    _write_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_01_build_skill_dataset_and_dspy import \
    SharedEvolutionObjects
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_03_run_gepa_uniform import \
    run_step as step_02_run_gepa_uniform
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_04_run_gepa_focused_on_difficulty import \
    run_step as step_03_run_gepa_focused_on_difficulty
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_05_run_gepa_gated import \
    run_step as step_04_run_gepa_gated
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_06_run_gepa_full import \
    run_step as step_05_run_gepa_full


class DemoTrainings:
    def __init__(self, config: DemoConfig):
        self._config: DemoConfig = config

    def run(
            self,
            params: DemoParams,
            baseline_score_single: float = None,
            baseline_score_multi: float = None,
            baseline_dims_multi = None,
            shared: SharedEvolutionObjects = None,
    ) -> DemoTrainingsResults:
        runs: list[tuple[str, object]] = []

        # ── gepa_uniform ─────────────────────────────────────────────────────────────
        scores_gepa_uniform, metrics_gepa_uniform, last_out_gepa_uniform = self._run_gepa_uniform(
            params, baseline_score_single,
            baseline_score_multi=baseline_score_multi, baseline_dims_multi=baseline_dims_multi,
            shared=shared,
        )
        if scores_gepa_uniform:
            runs.append(("GEPA-Uniform", last_out_gepa_uniform))

        # ── gepa_rubric ───────────────────────────────────────────────────────
        scores_gepa_rubric, metrics_gepa_rubric, last_out_gepa_rubric = self._run_gepa_rubric(
            params, baseline_score_single, baseline_score_multi=baseline_score_multi,
            baseline_dims_multi=baseline_dims_multi, shared=shared,
        )
        if scores_gepa_rubric:
            runs.append(("GEPA-Rubric", last_out_gepa_rubric))

        # ── gepa_focused_on_difficulty ───────────────────────────────────────────────────────────
        scores_gepa_focused, metrics_gepa_focused, last_out_gepa_focused = self._run_gepa_focused_on_difficulty(
            params, baseline_score_single,
            baseline_score_multi=baseline_score_multi, baseline_dims_multi=baseline_dims_multi,
            shared=shared,
        )
        if scores_gepa_focused:
            runs.append(("GEPA-Focused", last_out_gepa_focused))

        # ── gepa_gated ───────────────────────────────────────────────────────────
        scores_gepa_gated, metrics_gepa_gated, last_out_gepa_gated = self._run_gepa_gated(
            params, baseline_score_single,
            baseline_score_multi=baseline_score_multi, baseline_dims_multi=baseline_dims_multi,
            shared=shared,
        )
        if scores_gepa_gated:
            runs.append(("GEPA-Gated", last_out_gepa_gated))

        # ── gepa_full ─────────────────────────────────────────────────────────────
        scores_gepa_full, metrics_gepa_full, last_out_gepa_full = self._run_gepa_full(
            params, baseline_score_single,
            baseline_score_multi=baseline_score_multi, baseline_dims_multi=baseline_dims_multi,
            shared=shared,
        )
        if scores_gepa_full:
            runs.append(("GEPA-Full", last_out_gepa_full))

        return DemoTrainingsResults(
            runs=runs,
            scores_gepa_uniform=scores_gepa_uniform,
            scores_gepa_full=scores_gepa_full,
            scores_gepa_focused=scores_gepa_focused,
            scores_gepa_gated=scores_gepa_gated,
            scores_gepa_rubric=scores_gepa_rubric,
            metrics_gepa_uniform=metrics_gepa_uniform,
            metrics_gepa_full=metrics_gepa_full,
            metrics_gepa_focused=metrics_gepa_focused,
            metrics_gepa_gated=metrics_gepa_gated,
            metrics_gepa_rubric=metrics_gepa_rubric,
        )

    def _run_gepa_uniform(self, params: DemoParams, baseline_score: float = None,
                          baseline_score_multi: float = None, baseline_dims_multi=None,
                          shared: SharedEvolutionObjects = None):
        scores_gepa_uniform: list[float] = []
        metrics_gepa_uniform = None
        last_out_gepa_uniform = params.output_gepa_uniform
        if "gepa_uniform" in self._config.run_modes:
            t_start = time.monotonic()
            for i in range(1, self._config.n_runs + 1):
                out = self._out(params, "output_gepa_uniform", i)
                self._step_restore_baseline_skill(params)
                m = step_02_run_gepa_uniform(
                    params.skills_root, params.skill_name, self._config.model,
                    self._config.iterations, out, verbose=self._config.verbose,
                    baseline_score_single=baseline_score,
                    run_index=i, n_runs=self._config.n_runs,
                    prebuilt_skill=shared.skill if shared else None,
                    prebuilt_dataset=shared.dataset if shared else None,
                    prebuilt_baseline_module=shared.baseline_module if shared else None,
                    prebuilt_trainset=shared.trainset if shared else None,
                    prebuilt_valset=shared.valset if shared else None,
                )
                scores_gepa_uniform.append(m.get("evolved_score", 0.0))
                metrics_gepa_uniform = m
                last_out_gepa_uniform = out
            elapsed = time.monotonic() - t_start
            if len(scores_gepa_uniform) > 1:
                print_mode_summary("GEPA-Uniform", baseline_score, scores_gepa_uniform, elapsed_sec=elapsed)
            else:
                print_mode_timing("GEPA-Uniform", elapsed)
        return scores_gepa_uniform, metrics_gepa_uniform, last_out_gepa_uniform

    def _run_gepa_rubric(self, params: DemoParams, baseline_score_single: float = None,
                         baseline_score_multi: float = None, baseline_dims_multi=None,
                         shared: SharedEvolutionObjects = None):
        scores: list[float] = []
        metrics = None
        last_out = params.workdir / "output_gepa_rubric"
        if "gepa_rubric" in self._config.run_modes:
            t_start = time.monotonic()
            for i in range(1, self._config.n_runs + 1):
                out = self._out(params, "output_gepa_rubric", i)
                self._step_restore_baseline_skill(params)
                m = step_02_run_gepa_uniform(
                    params.skills_root, params.skill_name, self._config.model,
                    self._config.iterations, out, verbose=self._config.verbose,
                    baseline_score_single=baseline_score_single,
                    run_index=i, n_runs=self._config.n_runs,
                    scoring_mode="multi",
                    baseline_score_multi=baseline_score_multi,
                    baseline_dims_multi=baseline_dims_multi,
                    prebuilt_skill=shared.skill if shared else None,
                    prebuilt_dataset=shared.dataset if shared else None,
                    prebuilt_baseline_module=shared.baseline_module if shared else None,
                    prebuilt_trainset=shared.trainset if shared else None,
                    prebuilt_valset=shared.valset if shared else None,
                )
                scores.append(m.get("evolved_score", 0.0))
                metrics = m
                last_out = out
            elapsed = time.monotonic() - t_start
            if len(scores) > 1:
                print_mode_summary("GEPA-Rubric", baseline_score_single, scores, elapsed_sec=elapsed)
            else:
                print_mode_timing("GEPA-Rubric", elapsed)
        return scores, metrics, last_out

    def _run_gepa_full(self, params: DemoParams, baseline_score: float = None,
                      baseline_score_multi: float = None, baseline_dims_multi=None,
                      shared: SharedEvolutionObjects = None):
        scores_gepa_full: list[float] = []
        metrics_gepa_full = None
        last_out_gepa_full = params.output_gepa_full
        if "gepa_full" in self._config.run_modes:
            t_start = time.monotonic()
            for i in range(1, self._config.n_runs + 1):
                out = self._out(params, "output_gepa_full", i)
                ts  = self._ts(params, i)
                self._step_restore_baseline_skill(params)
                m = step_05_run_gepa_full(
                    params.skills_root, params.skill_name, self._config.model,
                    self._config.iterations, self._config.ts_batch_size, params.golden_examples,
                    out, ts, verbose=self._config.verbose,
                    baseline_score=baseline_score,
                    run_index=i, n_runs=self._config.n_runs,
                    prebuilt_skill=shared.skill if shared else None,
                    prebuilt_dataset=shared.dataset if shared else None,
                    prebuilt_baseline_module=shared.baseline_module if shared else None,
                    prebuilt_trainset=shared.trainset if shared else None,
                    prebuilt_valset=shared.valset if shared else None,
                )
                scores_gepa_full.append(m.get("evolved_score", 0.0))
                metrics_gepa_full = m
                last_out_gepa_full = out
            elapsed = time.monotonic() - t_start
            if len(scores_gepa_full) > 1:
                print_mode_summary("GEPA-Full", baseline_score, scores_gepa_full, elapsed_sec=elapsed)
            else:
                print_mode_timing("GEPA-Full", elapsed)
        return scores_gepa_full, metrics_gepa_full, last_out_gepa_full

    def _run_gepa_focused_on_difficulty(self, params: DemoParams, baseline_score: float = None,
                                        baseline_score_multi: float = None, baseline_dims_multi=None,
                                        shared: SharedEvolutionObjects = None):
        scores_gepa_focused: list[float] = []
        metrics_gepa_focused = None
        last_out_gepa_focused = params.output_gepa_focused_on_difficulty
        if "gepa_focused_on_difficulty" in self._config.run_modes:
            t_start = time.monotonic()
            for i in range(1, self._config.n_runs + 1):
                out = self._out(params, "output_gepa_focused_on_difficulty", i)
                ts  = self._ts(params, i)
                self._step_restore_baseline_skill(params)
                m = step_03_run_gepa_focused_on_difficulty(
                    params.skills_root, params.skill_name, self._config.model,
                    self._config.iterations, self._config.ts_batch_size,
                    out, ts, verbose=self._config.verbose,
                    baseline_score=baseline_score,
                    run_index=i, n_runs=self._config.n_runs,
                    prebuilt_skill=shared.skill if shared else None,
                    prebuilt_dataset=shared.dataset if shared else None,
                    prebuilt_baseline_module=shared.baseline_module if shared else None,
                    prebuilt_trainset=shared.trainset if shared else None,
                    prebuilt_valset=shared.valset if shared else None,
                )
                scores_gepa_focused.append(m.get("evolved_score", 0.0))
                metrics_gepa_focused = m
                last_out_gepa_focused = out
            elapsed = time.monotonic() - t_start
            if len(scores_gepa_focused) > 1:
                print_mode_summary("GEPA-Focused", baseline_score, scores_gepa_focused, elapsed_sec=elapsed)
            else:
                print_mode_timing("GEPA-Focused", elapsed)
        return scores_gepa_focused, metrics_gepa_focused, last_out_gepa_focused

    def _run_gepa_gated(self, params: DemoParams, baseline_score: float = None,
                       baseline_score_multi: float = None, baseline_dims_multi=None,
                       shared: SharedEvolutionObjects = None):
        scores_gepa_gated: list[float] = []
        metrics_gepa_gated = None
        last_out_gepa_gated = params.output_gepa_gated
        if "gepa_gated" in self._config.run_modes:
            t_start = time.monotonic()
            for i in range(1, self._config.n_runs + 1):
                out = self._out(params, "output_gepa_gated", i)
                ts  = self._ts(params, i)
                self._step_restore_baseline_skill(params)
                m = step_04_run_gepa_gated(
                    params.skills_root, params.skill_name, self._config.model,
                    self._config.iterations, out, ts, verbose=self._config.verbose,
                    baseline_score=baseline_score,
                    run_index=i, n_runs=self._config.n_runs,
                    prebuilt_skill=shared.skill if shared else None,
                    prebuilt_dataset=shared.dataset if shared else None,
                    prebuilt_baseline_module=shared.baseline_module if shared else None,
                    prebuilt_trainset=shared.trainset if shared else None,
                    prebuilt_valset=shared.valset if shared else None,
                )
                scores_gepa_gated.append(m.get("evolved_score", 0.0))
                metrics_gepa_gated = m
                last_out_gepa_gated = out
            elapsed = time.monotonic() - t_start
            if len(scores_gepa_gated) > 1:
                print_mode_summary("GEPA-Gated", baseline_score, scores_gepa_gated, elapsed_sec=elapsed)
            else:
                print_mode_timing("GEPA-Gated", elapsed)
        return scores_gepa_gated, metrics_gepa_gated, last_out_gepa_gated

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
