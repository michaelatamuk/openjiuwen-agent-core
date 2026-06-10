from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List


@dataclass
class GraphScore:
    """Concept-graph similarity scores for holdout evaluation.

    Dimensions
    ----------
    node_score — recall-biased F1 on concept nodes (unigrams + bigrams).
                 0.7 × recall + 0.3 × precision; rubric defines what must
                 be covered, so recall is weighted higher.
    edge_score — Jaccard similarity on co-occurrence edges (symmetric).
                 Captures relational structure: concepts that appear in the
                 same local context, not just the same document.

    Composite  — 0.6 × node_score + 0.4 × edge_score.

    No correctness gate is applied.  Graph similarity is a structural
    measure; the LLM holistic judge is the appropriate place to apply a
    semantic correctness gate.  Using graph scoring as a complementary
    signal alongside holistic/rubrics evaluation is the intended use case.
    """

    node_score: float = 0.0
    edge_score: float = 0.0

    DIM_NAMES: ClassVar[List[str]] = ["node_score", "edge_score"]

    @property
    def composite(self) -> float:
        return min(1.0, max(0.0, 0.6 * self.node_score + 0.4 * self.edge_score))

    def as_dict(self) -> Dict[str, float]:
        return {"node_score": self.node_score, "edge_score": self.edge_score}
