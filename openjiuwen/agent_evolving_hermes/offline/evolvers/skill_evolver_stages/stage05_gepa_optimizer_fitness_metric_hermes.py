from __future__ import annotations

import dspy


def fitness_metric(example: dspy.Example,
                   prediction: dspy.Prediction,
                   trace=None,
                   pred_name=None,
                   pred_trace=None) -> float:
    """Exact Hermes-style word-bag metric with 0.3 floor.

    score = 0.3 + 0.7 × (expected_words ∩ output_words) / |expected_words|

    The 0.3 floor ensures any non-empty output scores above zero even with
    no keyword overlap — matching the original Hermes behaviour.
    """
    if not getattr(prediction, "output", "").strip():
        return 0.0

    expected_words = set(example.expected_behavior.lower().split())
    output_words = set(prediction.output.lower().split())

    if not expected_words:
        return 0.5

    overlap = len(expected_words & output_words) / len(expected_words)
    return min(1.0, 0.3 + 0.7 * overlap)
