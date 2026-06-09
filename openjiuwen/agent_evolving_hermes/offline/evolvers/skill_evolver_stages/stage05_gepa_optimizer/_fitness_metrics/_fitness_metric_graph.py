from __future__ import annotations

from typing import FrozenSet, List, Set, Tuple

import dspy
from nltk.stem import PorterStemmer  # pip install nltk
from ._fitness_metric_stop_words import STOP_WORDS


# Initialize the stemmer once globally to avoid overhead inside the loop
_STEMMER = PorterStemmer()

# Sliding window size for edge construction
_WINDOW = 5


def _tokenize(text: str, stem = False) -> List[str]:
    """Lowercase, strip punctuation, stem, and return content tokens of length > 1."""
    tokens = []
    for w in text.lower().split():
        w = w.strip(".,!?;:\"'()[]{}\\/")
        if w and w not in STOP_WORDS:
            if stem:
                # Stem the word to reduce it to its base/root form
                stemmed_w = _STEMMER.stem(w)

                # Check length post-stemming to drop degenerate roots if any
                if len(stemmed_w) > 1:
                    tokens.append(stemmed_w)
            else:
                tokens.append(w)
    return tokens


def _concept_graph(text: str) -> Tuple[Set[str], Set[FrozenSet[str]]]:
    """Extract a concept graph from *text*.

    Nodes
    -----
    Stopword-filtered unigrams and bigrams.  Bigrams capture multi-word
    concepts (e.g. ``"database_connection"``, ``"error_handling"``) that
    would be split and diluted in a plain word-overlap metric.

    Edges
    -----
    Pairs of unigram nodes that co-occur within a sliding window of
    ``_WINDOW`` tokens.  This captures local relational structure: which
    concepts appear in the same context, serving as a proxy for semantic
    dependency without requiring a dependency parser.

    Returns
    -------
    nodes : set[str]
        Unigram and bigram concept tokens.
    edges : set[frozenset[str]]
        Unordered concept pairs (frozenset so order doesn't matter).
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


def _edge_str(e: FrozenSet[str]) -> str:
    a, b = sorted(e)
    return f"{a}↔{b}"


def fitness_metric(example: dspy.Example,
                   prediction: dspy.Prediction,
                   trace=None,
                   pred_name=None,
                   pred_trace=None):
    """Concept-graph structural similarity metric.

    Converts both ``example.expected_behavior`` (gold rubric) and
    ``prediction.output`` (agent response) into lightweight concept graphs,
    then measures their structural similarity.

    Graph representation
    --------------------
    Nodes  — stopword-filtered unigrams **and bigrams** of each text.
             Bigrams capture multi-word domain concepts that would be
             split and diluted by plain word-overlap metrics.
    Edges  — pairs of unigram nodes that co-occur within a sliding window
             of ``_WINDOW`` tokens.  These encode local relational structure
             (which concepts appear together) as a proxy for semantic
             dependency, without requiring an NLP parser.

    Score composition
    -----------------
    ::

        node_score = 0.7 × recall + 0.3 × precision   (recall-biased)
        edge_score = |exp_edges ∩ out_edges|            (Jaccard)
                     ─────────────────────────
                     |exp_edges ∪ out_edges|

        final = 0.6 × node_score + 0.4 × edge_score

    Node score is recall-biased because the rubric defines what must be
    covered; the agent must mention the right concepts.  Edge score uses
    Jaccard (symmetric) because structural overlap is a mutual property:
    GEPA benefits equally from detecting missing and spurious relational
    links.

    Why this differs from ``f1`` and ``bag_of_words``
    --------------------------------------------------
    A response that uses all the right keywords but scatters them across
    unrelated sentences will have the same node overlap as a well-structured
    response but lower edge overlap — the relational component penalises
    structural incoherence.  Conversely, a concise response that reproduces
    the key concept relationships (even with some paraphrasing) scores well
    on edges despite lower raw word overlap.

    GEPA feedback
    -------------
    When called from GEPA (``pred_name`` is not None), returns a
    ``dspy.Prediction(score, feedback)`` naming the specific concepts and
    concept-pairs that were missing.  The reflection LM can use this to
    propose targeted skill edits (e.g. add a step that requires the agent
    to address the missing relationship).  When called from MIPROv2 or the
    evaluation harness (``pred_name`` is None), returns a plain float for
    direct numeric aggregation.
    """
    if not getattr(prediction, "output", "").strip():
        if pred_name is not None:
            return dspy.Prediction(score=0.0, feedback="score=0.00; response was empty")
        return 0.0

    exp_nodes, exp_edges = _concept_graph(example.expected_behavior)
    out_nodes, out_edges = _concept_graph(prediction.output)

    if not exp_nodes:
        score = 0.5 if out_nodes else 0.0
        if pred_name is not None:
            return dspy.Prediction(score=score, feedback=f"score={score:.2f}; rubric had no content concepts to match")
        return score

    # ── Node similarity (recall-biased F1) ────────────────────────────────────
    node_isect = exp_nodes & out_nodes
    node_recall = len(node_isect) / len(exp_nodes)
    node_prec = len(node_isect) / len(out_nodes) if out_nodes else 0.0
    node_score = 0.7 * node_recall + 0.3 * node_prec

    # ── Edge similarity (Jaccard — symmetric structural overlap) ─────────────
    if exp_edges or out_edges:
        edge_isect = exp_edges & out_edges
        edge_union = exp_edges | out_edges
        edge_score = len(edge_isect) / len(edge_union)
        missing_edges = exp_edges - out_edges
    else:
        # Neither text had enough content tokens for edges; use node score only
        edge_score = node_score
        missing_edges = set()

    # ── Combined ──────────────────────────────────────────────────────────────
    score = min(1.0, max(0.0, 0.6 * node_score + 0.4 * edge_score))

    if pred_name is not None:
        missing_nodes = sorted(exp_nodes - out_nodes)[:6]
        missing_edge_strs = sorted(_edge_str(e) for e in missing_edges)[:4]
        parts = [f"score={score:.2f}"]
        if missing_nodes:
            parts.append(f"missing concepts: {', '.join(missing_nodes)}")
        else:
            parts.append("all concepts covered")
        if missing_edge_strs:
            parts.append(f"missing relations: {', '.join(missing_edge_strs)}")
        return dspy.Prediction(score=score, feedback="; ".join(parts))

    return score
