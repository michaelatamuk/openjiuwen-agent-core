
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_config import DemoConfig
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.demo_params import DemoParams
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.trainings.results import (
    DemoTrainingsResults,
    run_key_label,
)
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_mode_summary import \
    print_mode_summary, print_mode_timing
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.writer_skill import \
    _write_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.steps_shared_object import \
    SharedEvolutionObjects
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_03_run_gepa_plain import \
    run_step as step_03_run_gepa_plain
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_04_run_gepa_focused_on_difficulty import \
    run_step as _step_focused
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_05_run_gepa_gated import \
    run_step as _step_gated
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_06_run_gepa_full import \
    run_step as _step_full


class DemoTrainings:
    def __init__(self, config: DemoConfig):
        self._config: DemoConfig = config

    def run(
            self,
            params: DemoParams,
            shared_evolution_object: SharedEvolutionObjects,
            baseline_score_holistic: float = None,
            baseline_score_rubrics: float = None,
            baseline_dims_rubrics=None,
            console=None,
    ) -> DemoTrainingsResults:
        """Run every (mode, fitness_metric) combination and return aggregated results.

        For each mode in ``config.run_modes`` and each metric in
        ``config.fitness_metrics``:

        * When ``len(fitness_metrics) == 1``, the run key equals the mode
          name (``"gepa_plain_holistic"``), keeping output dirs identical to
          the single-metric behaviour.
        * When ``len(fitness_metrics) > 1``, the run key is
          ``"<mode>__<metric>"`` (e.g. ``"gepa_plain_holistic__jiuwen"``) and the
          output dir gets a matching suffix so every combination is stored
          independently.
        """
        fitness_metrics: List[str] = getattr(self._config, "fitness_metrics", ["jiuwen"]) or ["jiuwen"]
        multi_metric = len(fitness_metrics) > 1

        runs: List[Tuple[str, Path]] = []
        scores: Dict[str, List[float]] = {}
        metrics: Dict[str, Optional[dict]] = {}

        for mode in self._config.run_modes:
            for metric in fitness_metrics:
                run_key = f"{mode}__{metric}" if multi_metric else mode
                output_base = f"output_{run_key}"

                mode_scores, mode_metrics, last_out = self._run_mode_passes(
                    mode=mode,
                    metric=metric,
                    run_key=run_key,
                    output_base=output_base,
                    params=params,
                    shared=shared_evolution_object,
                    baseline_score_holistic=baseline_score_holistic,
                    baseline_score_rubrics=baseline_score_rubrics,
                    baseline_dims_rubrics=baseline_dims_rubrics,
                    console=console,
                )
                if mode_scores:
                    scores[run_key] = mode_scores
                    metrics[run_key] = mode_metrics
                    runs.append((run_key, last_out))

        return DemoTrainingsResults(runs=runs, scores=scores, metrics=metrics)

    # ── Core dispatcher ───────────────────────────────────────────────────

    def _run_mode_passes(
        self,
        mode: str,
        metric: str,
        run_key: str,
        output_base: str,
        params: DemoParams,
        shared: SharedEvolutionObjects,
        baseline_score_holistic: float,
        baseline_score_rubrics: float,
        baseline_dims_rubrics,
        console,
    ) -> Tuple[List[float], Optional[dict], Path]:
        """Run ``n_runs`` passes of *mode* with *metric*. Returns ``(scores, last_metrics, last_out_dir)``."""
        mode_scores: List[float] = []
        last_metrics: Optional[dict] = None
        last_out: Path = params.workdir / output_base

        t_start = time.monotonic()
        for i in range(1, self._config.n_runs + 1):
            output_dir = self._out(params, output_base, i)
            ts_state_dir = self._ts(params, output_base, i)
            self._step_restore_baseline_skill(params)

            m = self._dispatch(
                mode=mode,
                metric=metric,
                params=params,
                shared=shared,
                output_dir=output_dir,
                ts_state_dir=ts_state_dir,
                baseline_score_holistic=baseline_score_holistic,
                baseline_score_rubrics=baseline_score_rubrics,
                baseline_dims_rubrics=baseline_dims_rubrics,
                run_index=i,
                console=console,
            )
            mode_scores.append(m.get("evolved_score", 0.0))
            last_metrics = m
            last_out = output_dir

        elapsed = time.monotonic() - t_start
        label = run_key_label(run_key)
        if len(mode_scores) > 1:
            print_mode_summary(label, baseline_score_holistic, mode_scores,
                               elapsed_sec=elapsed, console=console)
        else:
            print_mode_timing(label, elapsed, console=console)

        return mode_scores, last_metrics, last_out

    def _dispatch(
        self,
        mode: str,
        metric: str,
        params: DemoParams,
        shared: SharedEvolutionObjects,
        output_dir: Path,
        ts_state_dir: Path,
        baseline_score_holistic: float,
        baseline_score_rubrics: float,
        baseline_dims_rubrics,
        run_index: int,
        console,
    ) -> dict:
        """Call the appropriate step function for *mode*."""
        common = dict(
            shared_evolution_object=shared,
            skills_root=params.skills_root,
            skill_name=params.skill_name,
            model=self._config.model,
            output_dir=output_dir,
            console=console,
            verbose=self._config.verbose,
            run_index=run_index,
            n_runs=self._config.n_runs,
            fitness_metric=metric,
        )

        if mode == "gepa_plain_holistic":
            return step_03_run_gepa_plain(
                **common,
                iterations=self._config.iterations,
                baseline_score_holistic=baseline_score_holistic,
                scoring_mode="holistic",
            ) or {}

        if mode == "gepa_plain_rubric":
            return step_03_run_gepa_plain(
                **common,
                iterations=self._config.iterations,
                baseline_score_holistic=baseline_score_holistic,
                scoring_mode="rubric",
                baseline_score_rubrics=baseline_score_rubrics,
                baseline_dims_rubrics=baseline_dims_rubrics,
            ) or {}

        if mode == "gepa_focused_on_difficulty":
            return _step_focused(
                **common,
                iterations=self._config.iterations,
                ts_batch_size=self._config.ts_batch_size,
                ts_state_dir=ts_state_dir,
                baseline_score_holistic=baseline_score_holistic,
            ) or {}

        if mode == "gepa_gated":
            return _step_gated(
                **common,
                iterations=self._config.iterations,
                ts_state_dir=ts_state_dir,
                baseline_score_holistic=baseline_score_holistic,
            ) or {}

        if mode == "gepa_full":
            return _step_full(
                **common,
                itrations=self._config.iterations,
                ts_batch_size=self._config.ts_batch_size,
                examples=params.golden_examples,
                ts_state_dir=ts_state_dir,
                baseline_score_holistic=baseline_score_holistic,
            ) or {}

        console.print(f"[yellow]Unknown mode '{mode}' — skipping[/yellow]")
        return {}

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _step_restore_baseline_skill(params: DemoParams) -> None:
        _write_skill(params.skills_root, params.skill_name, params.skill_frontmatter, params.skill_body)

    def _out(self, params: DemoParams, base: str, i: int) -> Path:
        """Return run-specific output dir; use canonical path for n_runs==1."""
        return params.workdir / base if self._config.n_runs == 1 else params.workdir / f"{base}_r{i}"

    def _ts(self, params: DemoParams, base: str, i: int) -> Path:
        """Return run-specific TS-state dir; suffixed by run_key to keep TS arms separate per combination."""
        if self._config.n_runs == 1:
            return params.workdir / f"ts_state_{base}"
        return params.workdir / f"ts_state_{base}_r{i}"
