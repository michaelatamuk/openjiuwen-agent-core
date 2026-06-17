# coding: utf-8
"""HuggingFace loader for GSM8K (Grade School Math 8K).

Dataset : openai/gsm8k  (main config, test split — 1,319 examples)
Paper   : OPRO arXiv:2309.03409, DSPy arXiv:2310.03714

The ``answer`` field already contains the full chain-of-thought ending with
``#### <number>``, so it is used verbatim as ``expected_behavior``.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List


def load(n: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Return ``n`` examples sampled from the GSM8K test split.

    Parameters
    ----------
    n:
        Number of examples to return.
    seed:
        Random seed for reproducible sampling.
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed)
    sample = ds.select(range(min(n, len(ds))))

    examples: List[Dict[str, Any]] = []
    for row in sample:
        examples.append(
            {
                "task_input": row["question"],
                "expected_behavior": row["answer"],
                "difficulty": "unknown",
                "source": "gsm8k-hf",
            }
        )
    return examples
