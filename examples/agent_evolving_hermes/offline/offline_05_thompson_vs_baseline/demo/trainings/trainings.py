
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
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_matrix_run_gepa import \
    run_step as step_matrix_run_gepa
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_matrix_save import \
    run_step as step_matrix_save
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
            baseline_score_graph: float = None,
            baseline_score_checklist: float = None,
            baseline_score_instruction_following: float = None,
            baseline_score_consistency: float = None,
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
        fitness_metrics: List[str] = getattr(self._config, "fitness_metrics", ["bag_of_words"]) or ["bag_of_words"]
        multi_metric = len(fitness_metrics) > 1

        runs: List[Tuple[str, Path]] = []
        scores: Dict[str, List[float]] = {}
        metrics: Dict[str, Optional[dict]] = {}

        for mode in self._config.run_modes:
            # ── gepa_scoring_matrix: handles all metrics internally ────────────
            if mode == "gepa_scoring_matrix":
                run_key = mode
                output_base = f"output_{mode}"
                _mode_scores: List[float] = []
                _last_metrics: Optional[dict] = None
                _last_out: Path = params.workdir / output_base

                _t_start = time.monotonic()
                for i in range(1, self._config.n_runs + 1):
                    _out_dir = self._out(params, output_base, i)
                    self._step_restore_baseline_skill(params)

                    _m = step_matrix_run_gepa(
                        shared_evolution_object=shared_evolution_object,
                        fitness_metrics=fitness_metrics,
                        skills_root=params.skills_root,
                        skill_name=params.skill_name,
                        model=self._config.model,
                        iterations=self._config.iterations,
                        output_dir=_out_dir,
                        console=console,
                        verbose=self._config.verbose,
                        baseline_score_holistic=baseline_score_holistic,
                        baseline_score_rubrics=baseline_score_rubrics,
                        baseline_dims_rubrics=baseline_dims_rubrics,
                        baseline_score_graph=baseline_score_graph,
                        baseline_score_checklist=baseline_score_checklist,
                        baseline_score_instruction_following=baseline_score_instruction_following,
                        baseline_score_consistency=baseline_score_consistency,
                        run_index=i,
                        n_runs=self._config.n_runs,
                    ) or {}

                    step_matrix_save(
                        matrix_summary=_m,
                        skill_name=params.skill_name,
                        output_dir=_out_dir,
                        model=self._config.model,
                        console=console,
                        oracle_data_dir=getattr(self._config, "oracle_data_dir", None),
                    )

                    _mode_scores.append(_m.get("evolved_score", 0.0))
                    _last_metrics = _m
                    _last_out = _out_dir

                _elapsed = time.monotonic() - _t_start
                _label = run_key_label(run_key)
                if len(_mode_scores) > 1:
                    print_mode_summary(_label, baseline_score_holistic, _mode_scores,
                                       elapsed_sec=_elapsed, console=console)
                else:
                    print_mode_timing(_label, _elapsed, console=console)

                if _mode_scores:
                    scores[run_key] = _mode_scores
                    metrics[run_key] = _last_metrics
                    runs.append((run_key, _last_out))
                continue
            # ──────────────────────────────────────────────────────────────────

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
                    baseline_score_graph=baseline_score_graph,
                    baseline_score_checklist=baseline_score_checklist,
                    baseline_score_instruction_following=baseline_score_instruction_following,
                    baseline_score_consistency=baseline_score_consistency,
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
        baseline_score_graph: float,
        baseline_score_checklist: float,
        baseline_score_instruction_following: float,
        baseline_score_consistency: float,
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
                baseline_score_graph=baseline_score_graph,
                baseline_score_checklist=baseline_score_checklist,
                baseline_score_instruction_following=baseline_score_instruction_following,
                baseline_score_consistency=baseline_score_consistency,
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
        baseline_score_graph: float,
        baseline_score_checklist: float,
        baseline_score_instruction_following: float,
        baseline_score_consistency: float,
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

        if mode == "gepa_plain_rubrics":
            return step_03_run_gepa_plain(
                **common,
                iterations=self._config.iterations,
                baseline_score_holistic=baseline_score_holistic,
                scoring_mode="rubrics",
                baseline_score_rubrics=baseline_score_rubrics,
                baseline_dims_rubrics=baseline_dims_rubrics,
            ) or {}

        if mode == "gepa_plain_graph":
            return step_03_run_gepa_plain(
                **common,
                iterations=self._config.iterations,
                baseline_score_holistic=baseline_score_holistic,
                scoring_mode="graph",
                baseline_score_graph=baseline_score_graph,
            ) or {}

        if mode == "gepa_plain_checklist":
            return step_03_run_gepa_plain(
                **common,
                iterations=self._config.iterations,
                baseline_score_holistic=baseline_score_holistic,
                scoring_mode="checklist",
                baseline_score_checklist=baseline_score_checklist,
            ) or {}

        if mode == "gepa_plain_instruction_following":
            return step_03_run_gepa_plain(
                **common,
                iterations=self._config.iterations,
                baseline_score_holistic=baseline_score_holistic,
                scoring_mode="instruction_following",
                baseline_score_instruction_following=baseline_score_instruction_following,
            ) or {}

        if mode == "gepa_plain_consistency":
            return step_03_run_gepa_plain(
                **common,
                iterations=self._config.iterations,
                baseline_score_holistic=baseline_score_holistic,
                scoring_mode="consistency",
                baseline_score_consistency=baseline_score_consistency,
            ) or {}

        if mode == "gepa_plain_comparative":
            return step_03_run_gepa_plain(
                **common,
                iterations=self._config.iterations,
                baseline_score_holistic=baseline_score_holistic,
                scoring_mode="comparative",
                # comparative uses fixed 0.5 neutral baseline — no prior_baseline_score needed
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
                iterations=self._config.iterations,
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
