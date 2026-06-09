from __future__ import annotations

from typing import Set

import dspy


# Common English stop words — filtered out during jiuwen F1 scoring
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "than", "that", "this", "these", "those", "it", "its", "also",
    "if", "then", "when", "where", "which", "who", "what", "how",
    "all", "any", "some", "such", "more", "most", "other", "same",
    "just", "about", "up", "out", "into", "through", "during",
}


def fitness_metric(example: dspy.Example,
                   prediction: dspy.Prediction,
                   trace=None,
                   pred_name=None,
                   pred_trace=None) -> float:
    """Stop-word-filtered weighted F1 metric for general-purpose skills.

    Removes common English stop words before computing overlap, then scores as:
        score = 0.7 × recall + 0.3 × precision  (on content words only)

    - Recall-heavy (0.7 weight): the agent must cover what the rubric expects.
    - Precision component (0.3 weight): rewards specificity, penalises irrelevant
      verbosity that happens to match by coincidence.
    - No artificial floor: zero overlap scores zero, giving GEPA and Thompson
      Sampling a clean signal for examples where the agent produces nothing useful.

    Works for any skill domain — not specific to code or technical content.
    """
    if not getattr(prediction, "output", "").strip():
        return 0.0

    def _content_words(text: str) -> Set[str]:
        return {w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 1}

    expected = _content_words(example.expected_behavior)
    output = _content_words(prediction.output)

    if not expected:
        return 0.5 if output else 0.0

    intersection = expected & output
    recall = len(intersection) / len(expected)
    precision = len(intersection) / len(output) if output else 0.0

    return min(1.0, max(0.0, 0.7 * recall + 0.3 * precision))
