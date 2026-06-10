from __future__ import annotations

from typing import FrozenSet, List, Set, Tuple

from ._stop_words import STOP_WORDS
from .score import GraphScore

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
    _NLTK_AVAILABLE = True
except ImportError:
    _STEMMER = None
    _NLTK_AVAILABLE = False

# Sliding window size for edge construction — must match the fitness metric
_WINDOW = 5


def _tokenize(text: str, stem: bool = False) -> List[str]:
    """Lowercase, strip punctuation, optionally stem; return content tokens of length > 1."""
    tokens = []
    for w in text.lower().split():
        w = w.strip(".,!?;:\"'()[]{}\\/")
        if not w or w in STOP_WORDS:
            continue
        if stem and _NLTK_AVAILABLE:
            w = _STEMMER.stem(w)
        if len(w) > 1:
            tokens.append(w)
    return tokens


def _concept_graph(text: str) -> Tuple[Set[str], Set[FrozenSet[str]]]:
    """Extract a concept graph from *text*.

    Nodes  — stopword-filtered unigrams and bigrams of content tokens.
    Edges  — pairs of unigram nodes that co-occur within ``_WINDOW`` tokens
             (proximity-based relational structure, no parser required).
    """
    tokens = _tokenize(text)
    if not tokens:
        return set(), set()

    unigrams: Set[str] = set(tokens)
    bigrams: Set[str] = {
        f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)
    }
    nodes = unigrams | bigrams

    edges: Set[FrozenSet[str]] = set()
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + _WINDOW, len(tokens))):
            if tokens[i] != tokens[j]:
                edges.add(frozenset({tokens[i], tokens[j]}))

    return nodes, edges


class GraphScorer:
    """Algorithmic concept-graph similarity scorer — no LLM calls.

    Mirrors the ``graph`` fitness metric used during GEPA optimisation so
    that the holdout signal lives in the same conceptual space as the inner
    loop.  Scoring is deterministic and fast (pure Python + optional NLTK
    stemming), making it a useful complement to the slower LLM judges.

    Usage
    -----
    ::

        scorer = GraphScorer()
        gs = scorer.score(example.expected_behavior, prediction.output)
        print(gs.composite, gs.node_score, gs.edge_score)
    """

    def score(self, expected_behavior: str, agent_output: str) -> GraphScore:
        """Return a :class:`GraphScore` for *agent_output* vs *expected_behavior*."""
        if not agent_output.strip():
            return GraphScore(node_score=0.0, edge_score=0.0)

        exp_nodes, exp_edges = _concept_graph(expected_behavior)
        out_nodes, out_edges = _concept_graph(agent_output)

        if not exp_nodes:
            fallback = 0.5 if out_nodes else 0.0
            return GraphScore(node_score=fallback, edge_score=fallback)

        # ── Node similarity (recall-biased F1) ────────────────────────────
        node_isect = exp_nodes & out_nodes
        node_recall = len(node_isect) / len(exp_nodes)
        node_prec = len(node_isect) / len(out_nodes) if out_nodes else 0.0
        node_score = 0.7 * node_recall + 0.3 * node_prec

        # ── Edge similarity (Jaccard) ─────────────────────────────────────
        if exp_edges or out_edges:
            edge_score = len(exp_edges & out_edges) / len(exp_edges | out_edges)
        else:
            edge_score = node_score  # no edges possible: fall back to node score

        return GraphScore(node_score=node_score, edge_score=edge_score)
