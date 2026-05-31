# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Eval dataset construction for GEPA skill evolution.

Mirrors hermes-agent-self-evolution evolution/core/dataset_builder.py exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dspy

from .eval_example import EvalExample


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
