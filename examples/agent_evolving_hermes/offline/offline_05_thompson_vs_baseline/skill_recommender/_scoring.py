# coding: utf-8
"""Shared text-overlap scoring utilities for benchmark loaders.

All loaders use these three metrics so scores are comparable across
benchmarks when they are mixed into the same oracle directory.
"""
from __future__ import annotations

import re
from collections import Counter

FITNESS_METRICS: list[str] = ["exact_match", "f1", "bag_of_words"]


def _normalise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def score_exact_match(output: str, expected: str) -> float:
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0


def score_f1(output: str, expected: str) -> float:
    pred = _normalise(output)
    gold = _normalise(expected)
    if not pred or not gold:
        return 0.0
    common   = Counter(pred) & Counter(gold)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred)
    recall    = n_common / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_bag_of_words(output: str, expected: str) -> float:
    pred = set(_normalise(output))
    gold = set(_normalise(expected))
    if not pred or not gold:
        return 0.0
    return len(pred & gold) / len(pred | gold)


_SCORERS: dict = {
    "exact_match":  score_exact_match,
    "f1":           score_f1,
    "bag_of_words": score_bag_of_words,
}


def compute_scores(output: str, expected: str) -> dict[str, float]:
    """Return {metric: score} for all FITNESS_METRICS."""
    return {m: fn(output, expected) for m, fn in _SCORERS.items()}
