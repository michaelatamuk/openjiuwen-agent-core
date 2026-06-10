from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List


@dataclass
class ConsistencyScore:
    """Output-consistency score for a repeated-run evaluation.

    Dimensions
    ----------
    mean_pairwise_similarity — mean word-level Jaccard similarity between
                               all pairs of outputs produced for the same
                               task input across ``n_samples`` runs.
                               1.0 = all outputs identical; 0.0 = no overlap.
    n_samples                — number of independent runs performed.

    Composite
    ---------
    Equal to ``mean_pairwise_similarity``.

    Interpretation
    --------------
    A consistency score near 1.0 means the skill produces stable, repeatable
    outputs.  A score near 0.0 means the skill is highly stochastic — the
    agent writes completely different things for the same input.  Neither
    extreme is universally good: some tasks benefit from diverse outputs;
    customer-facing skills usually require stability.

    Consistency is orthogonal to quality — a skill can be consistently
    wrong (high consistency, low holistic score) or inconsistently correct.
    """

    mean_pairwise_similarity: float = 0.0
    n_samples: int = 1

    DIM_NAMES: ClassVar[List[str]] = ["mean_pairwise_similarity"]

    @property
    def composite(self) -> float:
        return min(1.0, max(0.0, self.mean_pairwise_similarity))

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean_pairwise_similarity": self.mean_pairwise_similarity,
            "n_samples": float(self.n_samples),
        }
