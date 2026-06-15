"""Logging decorator for fitness metric functions.

Wraps any fitness metric callable and appends a record to *call_log* on
every invocation.  The metric's return value is passed through unchanged so
the GEPA optimizer sees exactly the same signal.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable, List


def _example_id(text: str) -> str:
    """8-char hex prefix of SHA-256 of *text* — stable key for grouping examples."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:8]


def wrap_metric_for_logging(metric_fn: Callable, call_log: List[dict]) -> Callable:
    """Return a wrapped version of *metric_fn* that logs every call.

    Parameters
    ----------
    metric_fn:
        Any fitness metric callable with the signature
        ``(example, prediction, trace=None, pred_name=None, pred_trace=None)``.
    call_log:
        A mutable list.  One dict is appended per invocation:
        ``{call_idx, example_id, example_input, example_expected, candidate_output, score, feedback}``.

    Returns
    -------
    Callable
        Drop-in replacement for *metric_fn*.  Same return type.
    """
    call_idx: List[int] = [0]

    def wrapped(example: Any, prediction: Any,
                trace: Any = None, pred_name: Any = None, pred_trace: Any = None) -> Any:
        result = metric_fn(example, prediction, trace, pred_name, pred_trace)

        # Normalise result — GEPA context returns dspy.Prediction(score, feedback),
        # plain eval context returns a raw float.
        if hasattr(result, "score"):
            score: float = float(result.score)
            feedback = getattr(result, "feedback", None)
        else:
            try:
                score = float(result)
            except (TypeError, ValueError):
                score = 0.0
            feedback = None

        ex_input = getattr(example, "task_input", "")
        call_log.append({
            "call_idx": call_idx[0],
            "example_id": _example_id(ex_input),
            "example_input": ex_input,
            "example_expected": getattr(example, "expected_behavior", ""),
            "candidate_output": getattr(prediction, "output", str(prediction)),
            "score": score,
            "feedback": feedback,
        })
        call_idx[0] += 1

        return result

    return wrapped
