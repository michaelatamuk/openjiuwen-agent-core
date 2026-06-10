from __future__ import annotations

from typing import List

import dspy
from ._fitness_metric_stop_words import STOP_WORDS


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stop words, return content tokens."""
    tokens = []
    for w in text.lower().split():
        w = w.strip(".,!?;:\"'()[]{}\\/")
        if w and w not in STOP_WORDS:
            tokens.append(w)
    return tokens


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Length of the Longest Common Subsequence of token lists (space-optimised DP)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Keep the shorter list as the inner dimension to minimise memory
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def fitness_metric(example: dspy.Example,
                   prediction: dspy.Prediction,
                   trace=None,
                   pred_name=None,
                   pred_trace=None):
    """ROUGE-L metric — Longest Common Subsequence F1.

    Computes the LCS F1 between stop-word-filtered tokens of
    ``example.expected_behavior`` and ``prediction.output``:

    ::

        recall    = LCS / |expected_tokens|
        precision = LCS / |output_tokens|
        ROUGE-L   = 2 × recall × precision / (recall + precision)

    Why ROUGE-L over ROUGE-1/2
    ---------------------------
    ROUGE-N requires *contiguous* n-gram matches; two texts can have
    identical unigrams but in completely different order.  LCS captures
    *order-preserving* subsequences, rewarding responses that follow the
    same logical progression as the rubric without requiring exact phrase
    matches.  This is useful for procedural / step-by-step skills.

    Why this differs from ``f1``
    ----------------------------
    ``f1`` (bag-of-words) is order-independent.  ROUGE-L penalises
    responses that cover all required keywords but in a different order
    than the rubric, which is a meaningful signal for sequential tasks
    (e.g. "list steps A, then B, then C").

    GEPA feedback
    -------------
    When called from GEPA (``pred_name`` is not None), returns a
    ``dspy.Prediction(score, feedback)`` reporting recall and precision
    so the reflection LM can diagnose whether the gap is coverage (low
    recall) or relevance (low precision).  When called from MIPROv2 or
    the evaluation harness (``pred_name`` is None), returns a plain float.
    """
    if not getattr(prediction, "output", "").strip():
        if pred_name is not None:
            return dspy.Prediction(score=0.0, feedback="score=0.00; response was empty")
        return 0.0

    exp_tokens = _tokenize(example.expected_behavior)
    out_tokens = _tokenize(prediction.output)

    if not exp_tokens:
        score = 0.5 if out_tokens else 0.0
        if pred_name is not None:
            return dspy.Prediction(
                score=score,
                feedback=f"score={score:.2f}; rubric had no content tokens to match",
            )
        return score

    lcs = _lcs_length(exp_tokens, out_tokens)

    if lcs == 0:
        if pred_name is not None:
            return dspy.Prediction(
                score=0.0,
                feedback="score=0.00; no common token subsequence found",
            )
        return 0.0

    recall = lcs / len(exp_tokens)
    precision = lcs / len(out_tokens) if out_tokens else 0.0
    score = (2.0 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0
    score = min(1.0, max(0.0, score))

    if pred_name is not None:
        parts = [
            f"score={score:.2f}",
            f"recall={recall:.2f}",
            f"precision={precision:.2f}",
            f"LCS={lcs}/{len(exp_tokens)} expected tokens",
        ]
        return dspy.Prediction(score=score, feedback="; ".join(parts))

    return score
