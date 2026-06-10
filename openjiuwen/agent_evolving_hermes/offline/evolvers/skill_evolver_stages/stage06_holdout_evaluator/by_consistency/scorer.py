from __future__ import annotations

from typing import List

import dspy

from .score import ConsistencyScore

_DEFAULT_N_SAMPLES = 3
_DEFAULT_TEMPERATURE = 0.7


def _word_jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class ConsistencyScorer:
    """Algorithmic consistency scorer — no LLM judge; uses the module itself.

    Runs ``n_samples`` independent forward passes for the same ``task_input``
    at ``temperature > 0``, then measures mean pairwise word-level Jaccard
    similarity across all output pairs.

    High similarity → stable, repeatable outputs.
    Low similarity  → stochastic, unpredictable outputs.

    Usage
    -----
    ::

        scorer = ConsistencyScorer()
        cs = scorer.score(module, ex.task_input, config.eval_model)
        print(cs.composite, cs.n_samples)

    Notes
    -----
    * Uses ``temperature=0.7`` by default.  Increase for more variance
      in the sample distribution; decrease to measure near-greedy stability.
    * Falls back to ``ConsistencyScore(1.0, 1)`` when fewer than 2 outputs
      are produced (degenerate case).
    """

    def __init__(self,
                 n_samples: int = _DEFAULT_N_SAMPLES,
                 temperature: float = _DEFAULT_TEMPERATURE):
        self.n_samples = n_samples
        self.temperature = temperature

    def score(self, module, task_input: str, model: str) -> ConsistencyScore:
        """Return a :class:`ConsistencyScore` for *module* on *task_input*."""
        outputs: List[str] = []
        lm = dspy.LM(model, temperature=self.temperature)
        with dspy.context(lm=lm):
            for _ in range(self.n_samples):
                try:
                    pred = module(task_input=task_input)
                    outputs.append(getattr(pred, "output", "") or "")
                except Exception:
                    outputs.append("")

        if len(outputs) < 2:
            return ConsistencyScore(
                mean_pairwise_similarity=1.0,
                n_samples=len(outputs),
            )

        similarities: List[float] = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                similarities.append(_word_jaccard(outputs[i], outputs[j]))

        mean_sim = sum(similarities) / len(similarities)
        return ConsistencyScore(
            mean_pairwise_similarity=mean_sim,
            n_samples=len(outputs),
        )
