# coding: utf-8
"""HuggingFace loader for HotPotQA (distractor setting).

Dataset : hotpotqa/hotpot_qa  (distractor config, validation split)
Paper   : DSPy arXiv:2310.03714

Uses streaming to avoid downloading the full ~500 MB split.  A buffer of
``buffer_size`` examples is fetched, then ``n`` are randomly sampled from it.

The dataset provides a ``level`` field ("easy" / "medium" / "hard") which is
mapped directly to ``difficulty``.
"""
from __future__ import annotations

import itertools
import random
from typing import Any, Dict, List


_DEFAULT_BUFFER = 500  # fetch this many streaming examples before sampling


def load(
    n: int = 50,
    seed: int = 42,
    buffer_size: int = _DEFAULT_BUFFER,
) -> List[Dict[str, Any]]:
    """Return ``n`` examples sampled from the HotPotQA validation split.

    Parameters
    ----------
    n:
        Number of examples to return.
    seed:
        Random seed for reproducible sampling.
    buffer_size:
        How many streaming examples to buffer before sampling ``n``.
        Larger values give a more representative sample but are slower.
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        streaming=True,
    )
    buffer = list(itertools.islice(ds, buffer_size))
    rng = random.Random(seed)
    sample = rng.sample(buffer, min(n, len(buffer)))

    examples: List[Dict[str, Any]] = []
    for row in sample:
        examples.append(
            {
                "task_input": row["question"],
                "expected_behavior": row["answer"],
                "difficulty": row.get("level", "unknown"),
                "source": "hotpotqa-hf",
            }
        )
    return examples
