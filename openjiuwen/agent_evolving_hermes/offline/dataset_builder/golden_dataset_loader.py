# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Eval dataset construction for GEPA skill evolution.

Mirrors hermes-agent-self-evolution evolution/core/dataset_builder.py exactly.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from .eval_dataset import EvalDataset
from .eval_example import EvalExample


class GoldenDatasetLoader:
    """Load a manually curated golden JSONL dataset.

    Mirrors Hermes GoldenDatasetLoader exactly:
      - If path is a directory with train.jsonl/val.jsonl/holdout.jsonl → load pre-split
      - If path is a single .jsonl file → auto-split 50/25/25

    This matches the Hermes evolution/core/dataset_builder.py implementation.
    """

    @staticmethod
    def load(path: Path) -> "EvalDataset":
        """Load golden dataset.  Handles both pre-split directories and single files."""
        path = Path(path)

        # ── Case 1: pre-split directory ──────────────────────────────────────
        if path.is_dir() and (path / "train.jsonl").exists():
            return EvalDataset.load(path)

        # ── Case 2: single JSONL file — auto-split (mirrors Hermes) ─────────
        golden_file: Optional[Path] = None
        if path.is_file() and path.suffix == ".jsonl":
            golden_file = path
        elif path.is_dir():
            candidate = path / "golden.jsonl"
            if candidate.exists():
                golden_file = candidate

        if golden_file is None:
            raise FileNotFoundError(
                f"No golden dataset found at {path}. "
                "Expected either a directory with train.jsonl or a single golden.jsonl file."
            )

        examples: List[EvalExample] = []
        with open(golden_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(EvalExample.from_dict(json.loads(line)))

        random.shuffle(examples)
        n = len(examples)
        n_train = max(1, int(n * 0.50))
        n_val = max(1, int(n * 0.25))
        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train: n_train + n_val],
            holdout=examples[n_train + n_val:],
        )
