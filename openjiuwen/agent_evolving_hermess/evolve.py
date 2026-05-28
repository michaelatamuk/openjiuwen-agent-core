# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Main GEPA evolution orchestration.

Mirrors hermes-agent-self-evolution evolve_skill.py.

New vs original plan:
  - Rich console output (Console, Panel, Table) — matches Hermess exactly
  - Dataset caching: if a dataset already exists for this skill, reuse it
    instead of regenerating (saves LLM cost on repeated runs)
  - Cross-run metrics history: each run appends to metrics_history.jsonl
    so regressions across runs are detectable
  - min_improvement acceptance gate: if improvement < threshold, the evolved
    skill is saved as evolved_REGRESSION.md and a warning is printed, but
    it is NOT written to evolved_skill.md (avoids regressing active skills)
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import dspy

from openjiuwen.agent_evolving_hermess.config import EvolverConfig
from openjiuwen.agent_evolving_hermess.constraints import ConstraintValidator
from openjiuwen.agent_evolving_hermess.dataset_builder import (
    EvalDataset,
    SyntheticDatasetBuilder,
)
from openjiuwen.agent_evolving_hermess.external_importers import build_dataset_from_external
from openjiuwen.agent_evolving_hermess.fitness import LLMJudge, skill_fitness_metric
from openjiuwen.agent_evolving_hermess.skill_module import (
    SkillModule,
    find_skill,
    load_skill,
    reassemble_skill,
)

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def _make_console() -> "Console":
    if _RICH:
        return Console()

    class _FallbackConsole:
        def print(self, *args, **kwargs):
            import re
            text = " ".join(str(a) for a in args)
            text = re.sub(r"\[/?[^\]]*\]", "", text)
            print(text)

        def rule(self, *a, **kw):
            print("-" * 60)

    return _FallbackConsole()  # type: ignore[return-value]


def _find_cached_dataset(skill_name: str, output_dir: Path) -> Optional[Path]:
    """Return the most recent cached dataset directory for this skill, or None."""
    skill_base = output_dir / skill_name
    if not skill_base.exists():
        return None
    candidates = sorted(
        (d / "dataset" for d in skill_base.iterdir() if d.is_dir()),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "train.jsonl").exists():
            return candidate
    return None


def _load_prior_metrics(skill_name: str, output_dir: Path) -> Optional[dict]:
    """Return the most recent metrics.json for this skill from a prior run.

    Scans timestamped run directories under ``output_dir / skill_name``,
    returning the metrics dict from the most recent run that has a
    ``metrics.json`` file.  Returns None if no prior runs exist.

    Used to detect cross-run regressions: compare the current run's
    ``baseline_score`` against the prior run's ``evolved_score`` to see
    whether a previously-evolved skill has deteriorated.
    """
    skill_base = output_dir / skill_name
    if not skill_base.exists():
        return None
    run_dirs = sorted(
        [d for d in skill_base.iterdir() if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        m = run_dir / "metrics.json"
        if m.exists():
            try:
                return json.loads(m.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def _append_metrics_history(skill_name: str, output_dir: Path, metrics: dict) -> None:
    """Append metrics to the per-skill metrics_history.jsonl file.

    This file accumulates one JSON record per run so regression trends
    can be plotted or scanned programmatically.  It lives at:
        <output_dir>/<skill_name>/metrics_history.jsonl
    """
    history_path = output_dir / skill_name / "metrics_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")


def evolve(
    skill_name: str,
    eval_source: str = "synthetic",
    external_sources: Optional[list] = None,
    iterations: Optional[int] = None,
    config: Optional[EvolverConfig] = None,
    reuse_dataset: bool = False,
    min_improvement: float = 0.0,
) -> dict:
    """Run one GEPA evolution pass on a skill.

    Mirrors hermes-agent-self-evolution evolve_skill.evolve() step by step.

    Args:
        skill_name: Name of the skill to evolve.
        eval_source: "synthetic" | "external" | "golden"
        external_sources: List of external log sources for eval_source="external".
        iterations: Override config.iterations if provided.
        config: EvolverConfig instance; defaults constructed if None.
        reuse_dataset: Reuse the most recently cached dataset (saves LLM cost).
        min_improvement: Minimum fitness improvement required before the evolved
            skill is accepted.  If improvement < min_improvement, the result is
            saved as evolved_REGRESSION.md but NOT deployed to evolved_skill.md.
            Use 0.0 (default) to accept any positive improvement.
            Use a negative value (e.g. -0.05) to accept minor regressions.

    Returns:
        Metrics dict including baseline_score, evolved_score, improvement,
        accepted (bool), and cross_run_delta (vs prior best if available).
    """
    if config is None:
        config = EvolverConfig()
    if iterations is not None:
        config.iterations = iterations

    console = _make_console()

    # ── Step 1: Find and load skill ──────────────────────────────────────────
    skill_path = find_skill(skill_name, config.skills_root)
    if skill_path is None:
        raise FileNotFoundError(
            f"Skill '{skill_name}' not found under {config.skills_root}"
        )
    skill = load_skill(skill_path)
    console.print(f"[bold]Loaded skill[/bold] '{skill['name']}' ({len(skill['raw'])} chars)")

    # Cross-run context: load prior run metrics for regression comparison
    prior_metrics = _load_prior_metrics(skill_name, config.output_dir)
    if prior_metrics:
        console.print(
            f"[dim]Prior run: baseline={prior_metrics['baseline_score']:.4f} "
            f"evolved={prior_metrics['evolved_score']:.4f} "
            f"({prior_metrics['timestamp']})[/dim]"
        )

    # ── Step 2: Validate baseline constraints ────────────────────────────────
    validator = ConstraintValidator(config)
    baseline_checks = validator.validate_all(skill["raw"], artifact_type="skill")
    failures = [c for c in baseline_checks if not c.passed]
    if failures:
        for f in failures:
            console.print(f"[red]✗ BASELINE CONSTRAINT FAILED: {f.constraint_name}: {f.message}[/red]")
        raise ValueError("Baseline skill fails constraints — fix before evolving.")
    console.print("[green]✓ Baseline constraints passed[/green]")

    # ── Step 3: Build / reuse eval dataset ──────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / skill_name / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"

    cached_path = _find_cached_dataset(skill_name, config.output_dir) if reuse_dataset else None

    if cached_path is not None:
        console.print(f"[cyan]Reusing cached dataset from {cached_path.parent.name}[/cyan]")
        dataset = EvalDataset.load(cached_path)
    elif eval_source == "synthetic":
        console.print("[bold]Building synthetic dataset…[/bold]")
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(skill["raw"], artifact_type="skill")
        dataset.save(dataset_dir)
    elif eval_source == "external":
        console.print("[bold]Building dataset from external session logs…[/bold]")
        sources = external_sources or ["jiuwen", "claude-code"]
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=sources,
            output_path=dataset_dir,
            model=config.eval_model,
        )
        if not dataset.train:
            console.print("[yellow]WARNING: No external examples — falling back to synthetic.[/yellow]")
            builder = SyntheticDatasetBuilder(config)
            dataset = builder.generate(skill["raw"], artifact_type="skill")
            dataset.save(dataset_dir)
    elif eval_source == "golden":
        golden_path = config.skills_root / skill_name / "golden_dataset"
        if not golden_path.exists():
            raise FileNotFoundError(f"No golden dataset at {golden_path}")
        from openjiuwen.agent_evolving_hermess.dataset_builder import GoldenDatasetLoader
        dataset = GoldenDatasetLoader.load(golden_path)
    else:
        raise ValueError(f"Unknown eval_source '{eval_source}'")

    console.print(
        f"Dataset: [bold]train={len(dataset.train)}[/bold] "
        f"val={len(dataset.val)} holdout={len(dataset.holdout)}"
    )
    if not dataset.train:
        raise ValueError("Empty training set — cannot evolve.")

    # ── Step 4: Configure DSPy + GEPA ────────────────────────────────────────
    dspy.configure(lm=dspy.LM(config.optimizer_model))
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")
    baseline_module = SkillModule(skill["raw"])

    # ── Step 5: Run GEPA ─────────────────────────────────────────────────────
    console.print(f"\n[bold]Running GEPA[/bold] ({config.iterations} iterations)…")
    t0 = time.time()
    optimizer_name = "GEPA"

    try:
        optimizer = dspy.GEPA(metric=skill_fitness_metric, max_steps=config.iterations)
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
    except Exception as gepa_err:
        console.print(f"[yellow]GEPA not available ({gepa_err}), falling back to MIPROv2[/yellow]")
        optimizer_name = "MIPROv2"
        optimizer = dspy.MIPROv2(metric=skill_fitness_metric, auto="light")
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)

    elapsed = time.time() - t0
    console.print(f"[green]✓ Optimisation complete in {elapsed:.1f}s[/green]")

    # ── Step 6: Extract evolved skill text ───────────────────────────────────
    evolved_body = optimized_module._skill_text_value
    evolved_text = reassemble_skill(skill["frontmatter_text"], evolved_body)

    # ── Step 7: Validate evolved constraints ─────────────────────────────────
    evolved_checks = validator.validate_all(
        evolved_text, artifact_type="skill", baseline_text=skill["raw"]
    )
    evolved_failures = [c for c in evolved_checks if not c.passed]
    if evolved_failures:
        failed_path = output_dir / "evolved_FAILED.md"
        failed_path.write_text(evolved_text)
        for f in evolved_failures:
            console.print(f"[red]✗ EVOLVED CONSTRAINT FAILED: {f.constraint_name}: {f.message}[/red]")
        console.print(f"[dim]Saved failed variant to {failed_path}[/dim]")
        raise ValueError("Evolved skill fails constraints — evolution rejected.")
    console.print("[green]✓ Evolved constraints passed[/green]")

    # ── Step 8: Evaluate on holdout ──────────────────────────────────────────
    judge = LLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)
    holdout = dataset.holdout or dataset.val

    def _eval_module(module: SkillModule, examples) -> float:
        scores = []
        for ex in examples:
            try:
                pred = module(task_input=ex.task_input)
                s = judge.score(
                    task_input=ex.task_input,
                    expected_behavior=ex.expected_behavior,
                    agent_output=getattr(pred, "output", ""),
                    skill_text=module._skill_text_value,
                )
                scores.append(s.composite)
            except Exception:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    console.print("[bold]Evaluating baseline on holdout…[/bold]")
    baseline_score = _eval_module(baseline_module, holdout)
    console.print("[bold]Evaluating evolved on holdout…[/bold]")
    evolved_score = _eval_module(optimized_module, holdout)
    improvement = evolved_score - baseline_score

    # Cross-run delta vs prior best
    cross_run_delta: Optional[float] = None
    if prior_metrics and "evolved_score" in prior_metrics:
        cross_run_delta = round(evolved_score - prior_metrics["evolved_score"], 4)

    # ── Step 9: min_improvement acceptance gate ───────────────────────────────
    accepted = improvement >= min_improvement
    if not accepted:
        regression_path = output_dir / "evolved_REGRESSION.md"
        regression_path.write_text(evolved_text)
        console.print(
            f"[yellow]⚠ Improvement {improvement:+.4f} < threshold {min_improvement:+.4f} "
            f"— not deploying (saved to evolved_REGRESSION.md)[/yellow]"
        )
        if cross_run_delta is not None:
            delta_color = "green" if cross_run_delta >= 0 else "red"
            console.print(
                f"[{delta_color}]Cross-run delta vs prior evolved: "
                f"{cross_run_delta:+.4f}[/{delta_color}]"
            )
    else:
        if improvement < 0:
            console.print(
                f"[yellow]⚠ Improvement {improvement:+.4f} is negative "
                f"(accepted because threshold is {min_improvement:+.4f})[/yellow]"
            )

    # ── Step 10: Display results table ───────────────────────────────────────
    if _RICH:
        table = Table(title="Evolution Results", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Skill", skill_name)
        table.add_row("Optimizer", optimizer_name)
        table.add_row("Iterations", str(config.iterations))
        table.add_row("Baseline score", f"{baseline_score:.4f}")
        table.add_row("Evolved score", f"{evolved_score:.4f}")
        sign = "+" if improvement >= 0 else ""
        color = "green" if improvement >= 0 else "red"
        table.add_row("Improvement", f"[{color}]{sign}{improvement:.4f}[/{color}]")
        if cross_run_delta is not None:
            xr_color = "green" if cross_run_delta >= 0 else "red"
            table.add_row(
                "Cross-run delta",
                f"[{xr_color}]{cross_run_delta:+.4f}[/{xr_color}]",
            )
        table.add_row("Accepted", "[green]YES[/green]" if accepted else "[red]NO[/red]")
        table.add_row("Elapsed", f"{elapsed:.1f}s")
        table.add_row("Baseline chars", str(len(skill["raw"])))
        table.add_row("Evolved chars", str(len(evolved_text)))
        console.print(table)
    else:
        console.print(
            f"\nHoldout: baseline={baseline_score:.3f} "
            f"evolved={evolved_score:.3f} "
            f"improvement={improvement:+.3f} "
            f"accepted={'YES' if accepted else 'NO'}"
        )
        if cross_run_delta is not None:
            console.print(f"Cross-run delta: {cross_run_delta:+.4f}")

    # ── Step 11: Save outputs ─────────────────────────────────────────────────
    metrics = {
        "skill_name": skill_name,
        "timestamp": ts,
        "baseline_score": round(baseline_score, 4),
        "evolved_score": round(evolved_score, 4),
        "improvement": round(improvement, 4),
        "accepted": accepted,
        "min_improvement_threshold": min_improvement,
        "cross_run_delta": cross_run_delta,
        "prior_run_timestamp": prior_metrics.get("timestamp") if prior_metrics else None,
        "iterations": config.iterations,
        "optimizer": optimizer_name,
        "eval_source": eval_source,
        "reused_dataset": cached_path is not None,
        "baseline_chars": len(skill["raw"]),
        "evolved_chars": len(evolved_text),
        "elapsed_seconds": round(elapsed, 1),
        "constraint_checks": [
            {"name": c.constraint_name, "passed": c.passed, "message": c.message}
            for c in evolved_checks
        ],
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Only deploy evolved_skill.md if the acceptance gate passed
    if accepted:
        (output_dir / "evolved_skill.md").write_text(evolved_text)
    (output_dir / "baseline_skill.md").write_text(skill["raw"])

    # Append to cross-run history log
    _append_metrics_history(skill_name, config.output_dir, metrics)

    console.print(f"\n[dim]Outputs saved to {output_dir}[/dim]")
    if accepted:
        console.print(f"[dim]History appended to {config.output_dir / skill_name / 'metrics_history.jsonl'}[/dim]")
    return metrics


def batch_evolve(
    skill_names: List[str],
    eval_source: str = "synthetic",
    external_sources: Optional[list] = None,
    iterations: Optional[int] = None,
    config: Optional[EvolverConfig] = None,
    reuse_dataset: bool = False,
    min_improvement: float = 0.0,
) -> List[dict]:
    """Evolve multiple skills sequentially.

    Returns a list of metrics dicts (one per skill).  If a skill fails,
    its entry contains ``{"skill_name": name, "error": "<message>"}``.

    Args:
        skill_names: List of skill names to evolve.
        (all other args: same as evolve())

    Returns:
        List of metrics dicts in the same order as skill_names.
    """
    results = []
    for name in skill_names:
        try:
            m = evolve(
                skill_name=name,
                eval_source=eval_source,
                external_sources=external_sources,
                iterations=iterations,
                config=config,
                reuse_dataset=reuse_dataset,
                min_improvement=min_improvement,
            )
        except Exception as exc:
            m = {"skill_name": name, "error": str(exc)}
        results.append(m)
    return results
