# coding: utf-8
"""HuggingFace loader for GSM8K (Grade School Math 8K).

Dataset : openai/gsm8k  (main config, test split — 1,319 examples)
Paper   : OPRO arXiv:2309.03409, DSPy arXiv:2310.03714

Dataset fetching is delegated to ``skill_recommender.gsm8k_loader.fetch_rows``
to avoid duplicating HuggingFace loading logic.

The ``answer`` field already contains the full chain-of-thought ending with
``#### <number>``, so it is used verbatim as ``expected_behavior``.
"""
from __future__ import annotations

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
    from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.gsm8k_loader import (
        fetch_rows,
    )

    return [
        {
            "task_input": row["question"],
            "expected_behavior": row["answer"],
            "difficulty": "unknown",
            "source": "gsm8k-hf",
        }
        for row in fetch_rows(n=n, seed=seed)
    ]
