# coding: utf-8
"""HuggingFace loader for PubMedQA (expert-labeled subset).

Dataset : qiaojin/PubMedQA  (pqa_labeled config, train split — 1,000 examples)
Paper   : SkillGen arXiv:2605.10999

``task_input`` is built by concatenating the abstract sentences (the
``context.contexts`` list) followed by the research question.

``expected_behavior`` is the ``final_decision`` verdict (yes / no / maybe)
followed by the expert ``long_answer``.  This matches the format the evolved
skill is expected to produce: verdict first, then evidence.
"""
from __future__ import annotations

import random
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
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ds = ds.shuffle(seed=seed)
    sample = ds.select(range(min(n, len(ds))))

    examples: List[Dict[str, Any]] = []
    for row in sample:
        context_sentences = row["context"]["contexts"]
        context_text = "\n".join(context_sentences)
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
