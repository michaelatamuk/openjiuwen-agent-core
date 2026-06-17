# coding: utf-8
"""HuggingFace loader for AQuA-RAT (Algebra Question Answering with Rationales).

Dataset : deepmind/aqua_rat  (raw config, test split — 254 examples)
Paper   : OPRO arXiv:2309.03409

``task_input`` formats the question together with the five lettered options.

``expected_behavior`` is the full ``rationale`` followed by
``Answer: <letter>``, matching the format the evolved skill must produce.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List


def load(n: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Return ``n`` examples sampled from the AQuA-RAT test split.

    Parameters
    ----------
    n:
        Number of examples to return.
    seed:
        Random seed for reproducible sampling.
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("deepmind/aqua_rat", "raw", split="test")
    ds = ds.shuffle(seed=seed)
    sample = ds.select(range(min(n, len(ds))))

    examples: List[Dict[str, Any]] = []
    for row in sample:
        options_text = "\n".join(row["options"])
        task_input = f"{row['question']}\nOptions:\n{options_text}"

        rationale = row.get("rationale", "").strip()
        correct = row.get("correct", "").strip()
        expected = f"{rationale}\nAnswer: {correct}" if rationale else f"Answer: {correct}"

        examples.append(
            {
                "task_input": task_input,
                "expected_behavior": expected,
                "difficulty": "unknown",
                "source": "aquarat-hf",
            }
        )
    return examples
