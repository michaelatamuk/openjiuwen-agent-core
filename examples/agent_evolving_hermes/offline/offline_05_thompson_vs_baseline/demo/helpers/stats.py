# coding: utf-8
"""Lightweight statistical helpers — no scipy dependency."""
from __future__ import annotations

import random
from typing import List, Tuple


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return *(mean, lower, upper)* bootstrap CI for the mean.

    Uses the percentile method.  Requires ``len(values) >= 2``.
    """
    if len(values) < 2:
        m = mean(values)
        return m, m, m
    rng = random.Random(seed)
    boot_means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(len(values))]
        boot_means.append(mean(sample))
    boot_means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[int((1.0 - alpha) * n_bootstrap)]
    return mean(values), lo, hi


def bootstrap_ci_diff(
    a: List[float],
    b: List[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for ``mean(b) - mean(a)`` (paired, same length).

    Returns *(mean_diff, lower, upper)*.  Significant improvement if
    the entire interval is above 0; significant regression if below 0.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    diffs = [bv - av for av, bv in zip(a, b)]
    return bootstrap_ci(diffs, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
