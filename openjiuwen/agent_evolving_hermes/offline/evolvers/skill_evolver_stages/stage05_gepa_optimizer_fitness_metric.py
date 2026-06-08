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

    These terms only appear in reviews that correctly identified the specific
    issue, making the metric much more discriminating than bag-of-words overlap.
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


def skill_fitness_metric(example: dspy.Example,
                         prediction: dspy.Prediction,
                         trace=None,
                         pred_name=None,
                         pred_trace=None) -> float:
    """Fast technical-keyword metric for GEPA's inner optimisation loop.

    Uses high-signal terms (backtick-wrapped tokens, ALL-CAPS acronyms,
    snake_case/dotted identifiers) extracted from expected_behavior rather
    than full bag-of-words overlap.

    Why this matters for Thompson Sampling:
    - Easy examples have common keywords (e.g. ``__main__``) that even the
      baseline skill mentions → low variance across evolved candidates → TS
      quickly deprioritises them.
    - Hard examples have rare, specific keywords (e.g. ``TOCTOU``,
      ``threading.Lock``, ``select_related``) that only a truly evolved skill
      mentions → high variance → TS focuses budget here.
    This creates the discriminating signal TS needs to outperform plain GEPA.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""

    if not agent_output.strip():
        return 0.0

    output_lower = agent_output.lower()

    tech_keywords = _extract_technical_keywords(expected)
    if tech_keywords:
        # Primary score: fraction of technical keywords present in output
        hits = sum(1 for kw in tech_keywords if kw in output_lower)
        tech_score = hits / len(tech_keywords)
        # Blend: 80% technical keyword coverage + 20% general word presence
        expected_words = set(expected.lower().split())
        output_words = set(output_lower.split())
        general_score = len(expected_words & output_words) / len(expected_words) if expected_words else 0.0
        score = 0.8 * tech_score + 0.2 * general_score
    else:
        # Fallback for examples with no extractable technical keywords.
        # No artificial floor: zero overlap must score zero so that TS arms
        # correctly accumulate β for examples where the agent produces nothing
        # useful.  The old 0.3 floor caused every no-keyword example to score
        # ≥ 0.3 regardless of output quality, inflating α and making TS
        # effectively random for those examples.
        expected_words = set(expected.lower().split())
        output_words = set(output_lower.split())
        overlap = len(expected_words & output_words) / len(expected_words) if expected_words else 0.0
        score = overlap

    return min(1.0, max(0.0, score))
