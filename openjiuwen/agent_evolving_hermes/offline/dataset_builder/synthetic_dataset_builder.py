# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Eval dataset construction for GEPA skill evolution.

Mirrors hermes-agent-self-evolution evolution/core/dataset_builder.py exactly.
"""
from __future__ import annotations

import json
import random
from typing import Optional

import dspy

from .eval_dataset import EvalDataset
from .eval_example import EvalExample


class SyntheticDatasetBuilder:
    """Generate eval dataset from skill text using LLM. Mirrors Hermes exactly."""

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
