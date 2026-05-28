# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Eval dataset construction for GEPA skill evolution.

Mirrors hermes-agent-self-evolution evolution/core/dataset_builder.py exactly.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import dspy


@dataclass
class EvalExample:
    task_input: str
    expected_behavior: str
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    category: str = "general"
    source: str = "synthetic"   # "synthetic" | "golden" | "jiuwen" | "claude-code"

    def to_dict(self) -> dict:
        return {
            "task_input": self.task_input,
            "expected_behavior": self.expected_behavior,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalExample":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class EvalDataset:
    train: List[EvalExample] = field(default_factory=list)
    val: List[EvalExample] = field(default_factory=list)
    holdout: List[EvalExample] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "holdout"):
            examples = getattr(self, split)
            with open(path / f"{split}.jsonl", "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        ds = cls()
        for split in ("train", "val", "holdout"):
            p = path / f"{split}.jsonl"
            if p.exists():
                examples = []
                with open(p) as f:
                    for line in f:
                        if line.strip():
                            examples.append(EvalExample.from_dict(json.loads(line)))
                setattr(ds, split, examples)
        return ds

    def to_dspy_examples(self, split: str = "train") -> list:
        return [
            dspy.Example(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
            ).with_inputs("task_input")
            for ex in getattr(self, split)
        ]


class GoldenDatasetLoader:
    """Load a manually curated golden JSONL dataset."""

    @staticmethod
    def load(path: Path) -> EvalDataset:
        """Load from a directory containing train.jsonl, val.jsonl, holdout.jsonl."""
        return EvalDataset.load(path)


class SyntheticDatasetBuilder:
    """Generate eval dataset from skill text using LLM. Mirrors Hermess exactly."""

    class GenerateTestCases(dspy.Signature):
        """Generate realistic evaluation test cases for a skill.

        Given the full text of a skill, generate diverse test cases
        that exercise different aspects of the skill. Each case must have:
        - task_input: a realistic user request (string)
        - expected_behavior: rubric describing a good response (string)
        - difficulty: "easy", "medium", or "hard"
        - category: which aspect of the skill this tests (string)
        """

        artifact_text: str = dspy.InputField(
            desc="Full text of the SKILL.md file"
        )
        artifact_type: str = dspy.InputField(
            desc="Type: 'skill', 'tool_description', or 'prompt_section'"
        )
        num_cases: int = dspy.InputField(desc="Number of test cases to generate")
        test_cases: str = dspy.OutputField(
            desc=(
                "JSON array of test cases, each with: "
                "task_input, expected_behavior, difficulty, category"
            )
        )

    def __init__(self, config: "EvolverConfig"):  # noqa: F821
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateTestCases)

    def generate(
        self,
        artifact_text: str,
        artifact_type: str = "skill",
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        n = num_cases or self.config.eval_dataset_size

        lm = dspy.LM(self.config.judge_model)
        with dspy.context(lm=lm):
            result = self.generator(
                artifact_text=artifact_text,
                artifact_type=artifact_type,
                num_cases=n,
            )

        try:
            cases_raw = json.loads(result.test_cases)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\[.*\]", result.test_cases, re.DOTALL)
            if m:
                cases_raw = json.loads(m.group())
            else:
                raise ValueError(
                    f"Could not parse test_cases JSON: {result.test_cases[:200]}"
                )

        examples = [
            EvalExample(
                task_input=c.get("task_input", ""),
                expected_behavior=c.get("expected_behavior", ""),
                difficulty=c.get("difficulty", "medium"),
                category=c.get("category", "general"),
                source="synthetic",
            )
            for c in cases_raw
            if c.get("task_input") and c.get("expected_behavior")
        ]

        random.shuffle(examples)
        n_total = len(examples)
        n_train = max(1, int(n_total * self.config.train_ratio))
        n_val = max(1, int(n_total * self.config.val_ratio))
        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train: n_train + n_val],
            holdout=examples[n_train + n_val:],
        )
