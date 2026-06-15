"""Step: run GEPA for every configured fitness metric with call-level logging.

For each fitness metric this step:
  1. Wraps the metric fn with a logging decorator that records every call.
  2. Runs a plain GEPA pass (step_03_run_gepa_plain) using the wrapped fn.
  3. Collects the per-call log and the run metrics into the matrix dict.

Returns a dict compatible with DemoTrainings' ``mode_scores`` collection:
  {
      "evolved_score":   float,   # mean evolved_score across all metrics
      "baseline_score":  float,   # mean baseline_score across all metrics
      "improvement":     float,
      "accepted":        bool,    # True if any metric run was accepted
      "matrix":          dict,    # full scoring matrix (passed to step_matrix_save)
  }
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.steps_shared_object import \
    SharedEvolutionObjects
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.steps.step_03_run_gepa_plain import \
    run_step as step_03_run_step
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage05_gepa_optimizer._fitness_metrics.fitness_metric_resolver import \
    resolve_fitness_metric
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage05_gepa_optimizer._fitness_metrics._fitness_metric_logging_wrapper import \
    wrap_metric_for_logging


def _infer_candidate_and_example_idx(call_log: list) -> None:
    """Annotate each call entry with ``candidate_idx`` and ``example_idx``.

    GEPA evaluates each candidate on a mini-batch of examples (often smaller
    than the full trainset).  We detect the batch boundary by watching when an
    ``example_input`` value that already appeared in the current batch shows up
    again — that restart signals a new candidate.

    Adds in-place:
      - ``candidate_idx``   : 0-based global candidate number
      - ``example_idx``     : 0-based position within that candidate's batch
      - ``_batch_size``     : inferred number of examples per candidate (for metadata)
    """
    if not call_log:
        return

    candidate_idx = 0
    seen_in_batch: list = []       # ordered list of example_inputs in current batch
    batch_size_samples: list = []  # collect observed batch sizes to pick the mode

    for entry in call_log:
        key = entry.get("example_input", "")

        if key in seen_in_batch:
            # Repeated input → new candidate batch starts
            batch_size_samples.append(len(seen_in_batch))
            candidate_idx += 1
            seen_in_batch = []

        example_idx = len(seen_in_batch)
        seen_in_batch.append(key)

        entry["candidate_idx"] = candidate_idx
        entry["example_idx"]   = example_idx

    # Record batch size from the last (possibly incomplete) batch
    if seen_in_batch:
        batch_size_samples.append(len(seen_in_batch))

    # Modal batch size (most common complete-batch size)
    inferred = max(set(batch_size_samples), key=batch_size_samples.count) if batch_size_samples else 0
    for entry in call_log:
        entry["_batch_size"] = inferred


def run_step(
    shared_evolution_object: SharedEvolutionObjects,
    fitness_metrics: List[str],
    skills_root,
    skill_name: str,
    model: str,
    iterations: int,
    output_dir: Path,
    console,
    verbose: bool = False,
    baseline_score_holistic: Optional[float] = None,
    baseline_score_rubrics: Optional[float] = None,
    baseline_dims_rubrics=None,
    baseline_score_graph: Optional[float] = None,
    baseline_score_checklist: Optional[float] = None,
    baseline_score_instruction_following: Optional[float] = None,
    baseline_score_consistency: Optional[float] = None,
    run_index: int = 1,
    n_runs: int = 1,
    custom_fitness_metrics: dict = None,
) -> dict:
    """Run GEPA once per fitness metric, logging every fitness call.

    Each metric gets its own sub-directory under *output_dir* so the individual
    evolved skills and metrics.json files do not overwrite each other.

    Returns
    -------
    dict
        Summary metrics dict compatible with ``DemoTrainings._run_mode_passes``,
        with an extra ``"matrix"`` key holding the full per-metric × per-call data.
    """
    console.print(f"\n[bold cyan]*** Demo Step (Matrix): Run GEPA Scoring Matrix Started ***[/bold cyan]")
    console.print(f"  Fitness metrics  : {', '.join(fitness_metrics)}")
    console.print(f"  Run index        : {run_index}/{n_runs}")

    custom_fitness_metrics = custom_fitness_metrics or {}
    matrix: dict = {}

    for metric_name in fitness_metrics:
        console.print(f"\n[bold yellow]  ── Matrix: running metric '{metric_name}' ──[/bold yellow]")

        # Per-metric output dir to avoid file-name collisions
        metric_out_dir = output_dir / f"metric_{metric_name}"
        metric_out_dir.mkdir(parents=True, exist_ok=True)

        # Build the logging wrapper
        call_log: list = []
        try:
            base_metric_fn = resolve_fitness_metric(metric_name, custom_fitness_metrics)
        except ValueError as exc:
            console.print(f"  [red]Cannot resolve metric '{metric_name}': {exc}[/red]")
            matrix[metric_name] = {
                "baseline_score": None,
                "evolved_score": None,
                "improvement": None,
                "calls": [],
                "error": str(exc),
            }
            continue

        wrapped_fn = wrap_metric_for_logging(base_metric_fn, call_log)

        # Run standard GEPA plain with the wrapped metric injected
        m = step_03_run_step(
            shared_evolution_object=shared_evolution_object,
            skills_root=skills_root,
            skill_name=skill_name,
            model=model,
            iterations=iterations,
            output_dir=metric_out_dir,
            console=console,
            verbose=verbose,
            baseline_score_holistic=baseline_score_holistic,
            run_index=run_index,
            n_runs=n_runs,
            scoring_mode="holistic",
            baseline_score_rubrics=baseline_score_rubrics,
            baseline_dims_rubrics=baseline_dims_rubrics,
            baseline_score_graph=baseline_score_graph,
            baseline_score_checklist=baseline_score_checklist,
            baseline_score_instruction_following=baseline_score_instruction_following,
            baseline_score_consistency=baseline_score_consistency,
            fitness_metric=metric_name,
            fitness_metric_fn_override=wrapped_fn,
        )

        # ── Enrich call log with structural indices ────────────────────────
        # GEPA sub-samples the trainset internally — it evaluates each candidate
        # on a mini-batch (commonly 3 examples), NOT the full trainset.
        # We detect the actual batch size dynamically: a new candidate starts
        # each time an example_input that was already seen in the current batch
        # appears again (i.e. the input sequence wraps).
        _infer_candidate_and_example_idx(call_log)

        n_examples = len(shared_evolution_object.trainset)
        inferred_batch_size = call_log[0].get("_batch_size", 1) if call_log else 0
        n_candidates = max((e["candidate_idx"] for e in call_log), default=-1) + 1 if call_log else 0

        matrix[metric_name] = {
            "baseline_score": m.get("baseline_score") if m else None,
            "evolved_score": m.get("evolved_score") if m else None,
            "improvement": m.get("improvement") if m else None,
            "accepted": m.get("accepted", False) if m else False,
            "n_examples_trainset": n_examples,
            "n_examples_per_candidate": inferred_batch_size,
            "n_candidates": n_candidates,
            "calls": call_log,
        }

    # ── Summary metrics for DemoTrainings integration ─────────────────────
    evolved_scores = [v["evolved_score"] for v in matrix.values() if v.get("evolved_score") is not None]
    baseline_scores = [v["baseline_score"] for v in matrix.values() if v.get("baseline_score") is not None]
    any_accepted = any(v.get("accepted", False) for v in matrix.values())

    mean_evolved = sum(evolved_scores) / len(evolved_scores) if evolved_scores else 0.0
    mean_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0

    summary = {
        "evolved_score": mean_evolved,
        "baseline_score": mean_baseline,
        "improvement": mean_evolved - mean_baseline,
        "accepted": any_accepted,
        "matrix": matrix,
        "fitness_metrics": fitness_metrics,
    }

    total_calls = sum(len(v["calls"]) for v in matrix.values())
    console.print(f"\n[bold cyan]*** Demo Step (Matrix): GEPA Scoring Matrix Finished ***[/bold cyan]")
    console.print(f"  Metrics run      : {len(matrix)}")
    console.print(f"  Mean evolved     : {mean_evolved:.4f}")
    console.print(f"  Total calls logged: {total_calls}")
    for mn, mv in matrix.items():
        n_calls = len(mv.get("calls", []))
        n_cands = mv.get("n_candidates", "?")
        batch   = mv.get("n_examples_per_candidate", "?")
        console.print(f"    {mn:<20} {n_calls:>4} calls  ({n_cands} candidates × {batch} examples/candidate)")

    return summary
