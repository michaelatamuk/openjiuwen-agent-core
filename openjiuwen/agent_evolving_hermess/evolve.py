# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Main GEPA evolution orchestration.

Mirrors hermes-agent-self-evolution evolve_skill.py.

New vs original plan:
  - Rich console output (Console, Panel, Table) — matches Hermess exactly
  - Dataset caching: if a dataset already exists for this skill, reuse it
    instead of regenerating (saves LLM cost on repeated runs)
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    from rich.panel import Panel
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def _make_console() -> "Console":
    if _RICH:
        return Console()
    # Fallback: minimal shim so the rest of the code works without rich
    class _FallbackConsole:
        def print(self, *args, **kwargs):
            # Strip rich markup tags for plain output
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
    # Scan timestamped run directories for a saved dataset
    candidates = sorted(
        (d / "dataset" for d in skill_base.iterdir() if d.is_dir()),
        key=lambda p: p.parent.name,  # sort by timestamp string
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "train.jsonl").exists():
            return candidate
    return None


def evolve(
    skill_name: str,
    eval_source: str = "synthetic",         # "synthetic" | "external" | "golden"
    external_sources: Optional[list] = None,
    iterations: Optional[int] = None,
    config: Optional[EvolverConfig] = None,
    reuse_dataset: bool = False,             # NEW: reuse cached dataset if available
) -> dict:
    """Run one GEPA evolution pass on a skill.

    Mirrors hermes-agent-self-evolution evolve_skill.evolve() step by step.

    Args:
        skill_name: Name of the skill to evolve.
        eval_source: "synthetic" | "external" | "golden"
        external_sources: List of external log sources for eval_source="external".
        iterations: Override config.iterations if provided.
        config: EvolverConfig instance; defaults constructed if None.
        reuse_dataset: If True, reuse the most recently cached dataset for this
            skill instead of generating a new one (saves LLM cost on re-runs).

    Returns:
        Metrics dict with baseline_score, evolved_score, improvement, paths.
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

    # Dataset caching: reuse the most recent saved dataset if requested
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
            console.print(
                "[yellow]WARNING: No external examples found — falling back to synthetic.[/yellow]"
            )
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

    # ── Step 5: Run GEPA ────────────────────────────────────────────────────
    console.print(f"\n[bold]Running GEPA[/bold] ({config.iterations} iterations)…")
    t0 = time.time()
    optimizer_name = "GEPA"

    try:
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric,
            max_steps=config.iterations,
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
    except Exception as gepa_err:
        console.print(f"[yellow]GEPA not available ({gepa_err}), falling back to MIPROv2[/yellow]")
        optimizer_name = "MIPROv2"
        optimizer = dspy.MIPROv2(
            metric=skill_fitness_metric,
            auto="light",
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )

    elapsed = time.time() - t0
    console.print(f"[green]✓ Optimisation complete in {elapsed:.1f}s[/green]")

    # ── Step 6: Extract evolved skill text ──────────────────────────────────
    evolved_body = optimized_module._skill_text_value
    evolved_text = reassemble_skill(skill["frontmatter_text"], evolved_body)

    # ── Step 7: Validate evolved constraints ────────────────────────────────
    evolved_checks = validator.validate_all(
        evolved_text, artifact_type="skill", baseline_text=skill["raw"]
    )
    evolved_failures = [c for c in evolved_checks if not c.passed]
    if evolved_failures:
        # Save failed variant for inspection (mirrors Hermess behavior)
        failed_path = output_dir / "evolved_FAILED.md"
        failed_path.write_text(evolved_text)
        for f in evolved_failures:
            console.print(
                f"[red]✗ EVOLVED CONSTRAINT FAILED: {f.constraint_name}: {f.message}[/red]"
            )
        console.print(f"[dim]Saved failed variant to {failed_path}[/dim]")
        raise ValueError("Evolved skill fails constraints — evolution rejected.")
    console.print("[green]✓ Evolved constraints passed[/green]")

    # ── Step 8: Evaluate on holdout ─────────────────────────────────────────
    judge = LLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)
    holdout = dataset.holdout or dataset.val  # fallback if holdout empty

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

    # ── Step 9: Display results table ────────────────────────────────────────
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
        table.add_row("Elapsed", f"{elapsed:.1f}s")
        table.add_row("Baseline chars", str(len(skill["raw"])))
        table.add_row("Evolved chars", str(len(evolved_text)))
        console.print(table)
    else:
        console.print(
            f"\nHoldout: baseline={baseline_score:.3f} "
            f"evolved={evolved_score:.3f} "
            f"improvement={improvement:+.3f}"
        )

    # ── Step 10: Save outputs ────────────────────────────────────────────────
    metrics = {
        "skill_name": skill_name,
        "timestamp": ts,
        "baseline_score": round(baseline_score, 4),
        "evolved_score": round(evolved_score, 4),
        "improvement": round(improvement, 4),
        "iterations": config.iterations,
        "optimizer": optimizer_name,
        "eval_source": eval_source,
        "reused_dataset": cached_path is not None,
        "baseline_chars": len(skill["raw"]),
        "evolved_chars": len(evolved_text),
        "elapsed_seconds": round(elapsed, 1),
        "constraint_checks": [
            {
                "name": c.constraint_name,
                "passed": c.passed,
                "message": c.message,
            }
            for c in evolved_checks
        ],
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "evolved_skill.md").write_text(evolved_text)
    (output_dir / "baseline_skill.md").write_text(skill["raw"])

    console.print(f"\n[dim]Outputs saved to {output_dir}[/dim]")
    return metrics
