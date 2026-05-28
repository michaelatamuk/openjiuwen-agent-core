# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Main GEPA evolution orchestration.

Mirrors hermes-agent-self-evolution evolve_skill.py.
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


def evolve(
    skill_name: str,
    eval_source: str = "synthetic",         # "synthetic" | "external" | "golden"
    external_sources: Optional[list] = None,
    iterations: Optional[int] = None,
    config: Optional[EvolverConfig] = None,
) -> dict:
    """Run one GEPA evolution pass on a skill.

    Mirrors hermes-agent-self-evolution evolve_skill.evolve() step by step.

    Returns metrics dict with baseline_score, evolved_score, improvement, paths.
    """
    if config is None:
        config = EvolverConfig()
    if iterations is not None:
        config.iterations = iterations

    # ── Step 1: Find and load skill ──────────────────────────────────────────
    skill_path = find_skill(skill_name, config.skills_root)
    if skill_path is None:
        raise FileNotFoundError(
            f"Skill '{skill_name}' not found under {config.skills_root}"
        )
    skill = load_skill(skill_path)
    print(f"[evolve] Loaded skill '{skill['name']}' ({len(skill['raw'])} chars)")

    # ── Step 2: Validate baseline constraints ────────────────────────────────
    validator = ConstraintValidator(config)
    baseline_checks = validator.validate_all(skill["raw"], artifact_type="skill")
    failures = [c for c in baseline_checks if not c.passed]
    if failures:
        for f in failures:
            print(f"[evolve] BASELINE CONSTRAINT FAILED: {f.constraint_name}: {f.message}")
        raise ValueError("Baseline skill fails constraints — fix before evolving.")

    # ── Step 3: Build eval dataset ───────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / skill_name / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"

    if eval_source == "synthetic":
        print("[evolve] Building synthetic dataset...")
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(skill["raw"], artifact_type="skill")
        dataset.save(dataset_dir)
    elif eval_source == "external":
        print("[evolve] Building dataset from external session logs...")
        sources = external_sources or ["jiuwen", "claude-code"]
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=sources,
            output_path=dataset_dir,
            model=config.eval_model,
        )
        if not dataset.train:
            print("[evolve] WARNING: No external examples found; falling back to synthetic.")
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

    print(
        f"[evolve] Dataset: train={len(dataset.train)} "
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
    print(f"[evolve] Running GEPA ({config.iterations} iterations)...")
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
        print(f"[evolve] GEPA failed ({gepa_err}); falling back to MIPROv2...")
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
    print(f"[evolve] Optimisation complete in {elapsed:.1f}s")

    # ── Step 6: Extract evolved skill text ──────────────────────────────────
    evolved_body = optimized_module._skill_text_value
    evolved_text = reassemble_skill(skill["frontmatter_text"], evolved_body)

    # ── Step 7: Validate evolved constraints ────────────────────────────────
    evolved_checks = validator.validate_all(
        evolved_text, artifact_type="skill", baseline_text=skill["raw"]
    )
    evolved_failures = [c for c in evolved_checks if not c.passed]
    if evolved_failures:
        for f in evolved_failures:
            print(f"[evolve] EVOLVED CONSTRAINT FAILED: {f.constraint_name}: {f.message}")
        raise ValueError("Evolved skill fails constraints — evolution rejected.")

    # ── Step 8: Evaluate on holdout ─────────────────────────────────────────
    judge = LLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)
    holdout = dataset.holdout or dataset.val  # fallback if holdout is empty

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

    print("[evolve] Evaluating baseline on holdout...")
    baseline_score = _eval_module(baseline_module, holdout)
    print("[evolve] Evaluating evolved on holdout...")
    evolved_score = _eval_module(optimized_module, holdout)

    improvement = evolved_score - baseline_score
    print(
        f"[evolve] Holdout: baseline={baseline_score:.3f} "
        f"evolved={evolved_score:.3f} "
        f"improvement={improvement:+.3f}"
    )

    # ── Step 9: Save outputs ─────────────────────────────────────────────────
    metrics = {
        "skill_name": skill_name,
        "timestamp": ts,
        "baseline_score": round(baseline_score, 4),
        "evolved_score": round(evolved_score, 4),
        "improvement": round(improvement, 4),
        "iterations": config.iterations,
        "optimizer": optimizer_name,
        "eval_source": eval_source,
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

    print(f"[evolve] Outputs saved to {output_dir}")
    return metrics
