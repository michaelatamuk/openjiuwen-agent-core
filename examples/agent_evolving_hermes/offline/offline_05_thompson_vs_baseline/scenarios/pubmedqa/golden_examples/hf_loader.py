# coding: utf-8
"""HuggingFace loader for PubMedQA (expert-labeled subset).

Dataset : qiaojin/PubMedQA  (pqa_labeled config, train split — 1,000 examples)
Paper   : SkillGen arXiv:2605.10999

Dataset fetching is delegated to ``skill_recommender.pubmedqa_loader.fetch_rows``
to avoid duplicating HuggingFace loading logic.

``task_input`` is built by concatenating the abstract sentences (the
``context.contexts`` list) followed by the research question.

``expected_behavior`` is the ``final_decision`` verdict (yes / no / maybe)
followed by the expert ``long_answer``.  This matches the format the evolved
skill is expected to produce: verdict first, then evidence.
"""
from __future__ import annotations

from typing import Any, Dict, List


def load(n: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """Return ``n`` examples sampled from PubMedQA pqa_labeled train split.

    Parameters
    ----------
    n:
        Number of examples to return.
    seed:
        Random seed for reproducible sampling.
    """
    from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.pubmedqa_loader import (
        fetch_rows,
    )

    examples: List[Dict[str, Any]] = []
    for row in fetch_rows(n=n, seed=seed):
        context_text = "\n".join(row["context"]["contexts"])
        task_input = f"Context:\n{context_text}\n\nQuestion: {row['question']}"

        verdict = row["final_decision"]
        long_answer = row.get("long_answer", "").strip()
        expected = f"{verdict}\n{long_answer}" if long_answer else verdict

        examples.append(
            {
                "task_input": task_input,
                "expected_behavior": expected,
                "difficulty": "unknown",
                "source": "pubmedqa-hf",
            }
        )
    return examples
