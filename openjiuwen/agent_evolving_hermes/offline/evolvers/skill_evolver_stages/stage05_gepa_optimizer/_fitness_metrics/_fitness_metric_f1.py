from __future__ import annotations

from typing import Set

import dspy
from ._fitness_metric_stop_words import STOP_WORDS


def fitness_metric(example: dspy.Example,
                   prediction: dspy.Prediction,
                   trace=None,
                   pred_name=None,
                   pred_trace=None):
    """Stop-word-filtered weighted F1 metric for general-purpose skills.

    Removes common English stop words before computing overlap, then scores as:
        score = 0.7 × recall + 0.3 × precision  (on content words only)

    - Recall-heavy (0.7 weight): the agent must cover what the rubric expects.
    - Precision component (0.3 weight): rewards specificity, penalises irrelevant
      verbosity that happens to match by coincidence.
    - No artificial floor: zero overlap scores zero, giving GEPA and Thompson
      Sampling a clean signal for examples where the agent produces nothing useful.

    Works for any skill domain — not specific to code or technical content.

    When called from GEPA (``pred_name`` is not None), returns a
    ``dspy.Prediction(score, feedback)`` so the reflection LM receives
    actionable context: which rubric concepts were missing (recall gap) and
    whether the response contained irrelevant content (precision loss).  When
    called from MIPROv2 or the evaluation harness (``pred_name`` is None),
    returns a plain float for direct numeric aggregation.
    """
    if not getattr(prediction, "output", "").strip():
        if pred_name is not None:
            return dspy.Prediction(score=0.0, feedback="score=0.00; response was empty")
        return 0.0

    def _content_words(text: str) -> Set[str]:
        return {w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 1}

    expected = _content_words(example.expected_behavior)
    output = _content_words(prediction.output)

    if not expected:
        score = 0.5 if output else 0.0
        if pred_name is not None:
            return dspy.Prediction(score=score, feedback=f"score={score:.2f}; rubric had no content words to match")
        return score

    intersection = expected & output
    recall = len(intersection) / len(expected)
    precision = len(intersection) / len(output) if output else 0.0
    score = min(1.0, max(0.0, 0.7 * recall + 0.3 * precision))

    if pred_name is not None:
        missing = sorted(expected - output)[:6]
        parts = [f"score={score:.2f}"]
        if missing:
            parts.append(f"missing key concepts: {', '.join(missing)}")
        else:
            parts.append("all key concepts covered")
        # Only mention precision loss if it's significantly dragging the score
        if precision < 0.4 and output:
            irrelevant = sorted(output - expected)[:4]
            if irrelevant:
                parts.append(f"off-topic terms: {', '.join(irrelevant)}")
        return dspy.Prediction(score=score, feedback="; ".join(parts))

    return score
