# coding: utf-8
"""Custom fitness metric example: technical-keyword scoring.

This module provides a fitness metric that gives extra weight to high-signal
technical terms found in the expected_behavior rubric: backtick-wrapped tokens,
ALL-CAPS acronyms, and snake_case/dotted identifiers.

Originally the default metric in the jiuwen GEPA optimizer.  Moved here as an
example of how to supply a ``custom`` fitness metric for skill domains that
benefit from emphasising specific technical vocabulary.

Usage in config
---------------
Point ``fitness_metric`` at this function's dotted import path:

    "fitness_metric": "examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.custom_fitness_metric_tech_keywords.tech_keyword_fitness_metric"

Or register it under a short name:

    evolver_config = EvolverConfig(
        fitness_metric="tech",
        custom_fitness_metrics={
            "tech": tech_keyword_fitness_metric,
        },
    )

Why technical keywords?
-----------------------
- Easy examples have common keywords (e.g. ``__main__``) that even the baseline
  skill mentions → low variance across evolved candidates → Thompson Sampling (TS)
  quickly deprioritises them.
- Hard examples have rare, specific keywords (e.g. ``TOCTOU``, ``threading.Lock``,
  ``select_related``) that only a truly evolved skill mentions → high variance → TS
  focuses budget here.
- Blend: 80 % technical keyword coverage + 20 % general word presence.
- Fallback to pure word-overlap (no artificial floor) when no technical keywords
  are extractable, so TS arms accumulate β correctly for those examples.

Caveats
-------
This metric is domain-specific: it works best for technical code-review or
security-analysis skills where the rubric contains backtick-wrapped identifiers
and ALL-CAPS acronyms.  For non-technical skills prefer the built-in
``"bag_of_words"`` metric.
"""
from __future__ import annotations

import re
from typing import Set

import dspy


def _extract_technical_keywords(text: str) -> Set[str]:
    """Extract high-signal technical terms from an expected_behavior string.

    Three categories, in descending specificity:
      1. Backtick-wrapped terms  → ``threading.Lock``, ``select_related('product')``
      2. ALL-CAPS acronyms       → TOCTOU, SQL, N+1, O(N)
      3. snake_case / dotted identifiers with ≥2 parts → os.path.exists, page_num
    """
    keywords: Set[str] = set()

    # 1. Backtick-wrapped (keep full token, e.g. "threading.lock", "items=none")
    for m in re.finditer(r"`([^`]+)`", text):
        kw = m.group(1).lower().strip()
        if kw:
            keywords.add(kw)

    # 2. ALL-CAPS acronyms and patterns like N+1, O(N)
    for m in re.finditer(r"\b([A-Z]{2,}(?:[+\-*/()\d]*)?)\b", text):
        keywords.add(m.group(1).lower())

    # 3. snake_case or dotted names with ≥2 parts (e.g. page_num, os.path)
    for m in re.finditer(r"\b([a-z][a-z0-9]*(?:[._][a-z][a-z0-9]*)+)\b", text):
        keywords.add(m.group(1).lower())

    return keywords


def tech_keyword_fitness_metric(example: dspy.Example,
                          prediction: dspy.Prediction,
                          trace=None,
                          pred_name=None,
                          pred_trace=None) -> float:
    """Fast technical-keyword metric for GEPA's inner optimization loop.

    Uses high-signal terms (backtick-wrapped tokens, ALL-CAPS acronyms,
    snake_case/dotted identifiers) extracted from expected_behavior rather
    than full bag-of-words overlap.
    """
    if not getattr(prediction, "output", "").strip():
        return 0.0

    expected_words = set(example.expected_behavior.lower().split())
    output_words = set(prediction.output.lower().split())

    tech_keywords = _extract_technical_keywords(example.expected_behavior)
    if tech_keywords:
        hits = sum(1 for kw in tech_keywords if kw in prediction.output.lower())
        tech_score = hits / len(tech_keywords)
        general_score = len(expected_words & output_words) / len(expected_words) if expected_words else 0.0
        return min(1.0, max(0.0, 0.8 * tech_score + 0.2 * general_score))
    else:
        if not expected_words:
            return 0.0
        return min(1.0, max(0.0, len(expected_words & output_words) / len(expected_words)))
