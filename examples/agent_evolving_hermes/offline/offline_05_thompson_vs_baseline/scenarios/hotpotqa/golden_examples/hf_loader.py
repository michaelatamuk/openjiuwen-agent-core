# coding: utf-8
"""HuggingFace loader for HotPotQA (distractor setting).

Dataset : hotpotqa/hotpot_qa  (distractor config, validation split)
Paper   : DSPy arXiv:2310.03714

Dataset fetching is delegated to ``skill_recommender.hotpotqa_loader.fetch_rows``
to avoid duplicating HuggingFace loading logic.

The dataset's ``level`` field (easy / medium / hard) is mapped directly to
``difficulty``.
"""
from __future__ import annotations

from typing import Any, Dict, List


def load(n: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Return ``n`` examples sampled from the HotPotQA validation split.

    Parameters
    ----------
    n:
        Number of examples to return.
    seed:
        Random seed for reproducible sampling.
    """
    from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.hotpotqa_loader import (
        fetch_rows,
    )

    return [
        {
            "task_input": row["question"],
            "expected_behavior": row["answer"],
            "difficulty": row.get("level", "unknown"),
            "source": "hotpotqa-hf",
        }
        for row in fetch_rows(n=n, seed=seed)
    ]
