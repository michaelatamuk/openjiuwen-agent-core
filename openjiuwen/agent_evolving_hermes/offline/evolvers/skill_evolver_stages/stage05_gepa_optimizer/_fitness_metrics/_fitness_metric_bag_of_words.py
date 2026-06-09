from __future__ import annotations

import dspy


def fitness_metric(example: dspy.Example,
                   prediction: dspy.Prediction,
                   trace=None,
                   pred_name=None,
                   pred_trace=None):
    """Exact Hermes-style word-bag metric with 0.3 floor.

    score = 0.3 + 0.7 × (expected_words ∩ output_words) / |expected_words|

    The 0.3 floor ensures any non-empty output scores above zero even with
    no keyword overlap — matching the original Hermes behaviour.

    When called from GEPA (``pred_name`` is not None), returns a
    ``dspy.Prediction(score, feedback)`` so the reflection LM receives
    actionable context about which rubric terms were missing.  When called
    from MIPROv2 or the evaluation harness (``pred_name`` is None), returns
    a plain float for direct numeric aggregation.
    """
    if not getattr(prediction, "output", "").strip():
        if pred_name is not None:
            return dspy.Prediction(score=0.0, feedback="score=0.00; response was empty")
        return 0.0

    expected_words = set(example.expected_behavior.lower().split())
    output_words = set(prediction.output.lower().split())

    if not expected_words:
        score = 0.5
        if pred_name is not None:
            return dspy.Prediction(score=score, feedback=f"score={score:.2f}; rubric had no keywords to match")
        return score

    overlap = len(expected_words & output_words) / len(expected_words)
    score = min(1.0, 0.3 + 0.7 * overlap)

    if pred_name is not None:
        missing = sorted(expected_words - output_words)[:6]
        parts = [f"score={score:.2f}"]
        if missing:
            parts.append(f"missing rubric terms: {', '.join(missing)}")
        else:
            parts.append("all rubric terms covered")
        return dspy.Prediction(score=score, feedback="; ".join(parts))

    return score
