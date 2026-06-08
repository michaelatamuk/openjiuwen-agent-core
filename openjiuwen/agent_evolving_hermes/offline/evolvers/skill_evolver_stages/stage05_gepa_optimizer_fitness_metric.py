from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Set

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


def hermes_fitness_metric(example: dspy.Example,
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


def jiuwen_fitness_metric(example: dspy.Example,
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


def _ensure_gepa_signature(fn: Callable) -> Callable:
    """Wrap *fn* so it always accepts exactly 5 positional arguments.

    GEPA inspects the metric signature and requires ``(gold, pred, trace,
    pred_name, pred_trace)``.  Built-in metrics already carry that signature;
    this wrapper ensures custom / imported metrics are compatible too.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
        positional_params = [
            p for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional_params) >= 5:
            return fn  # already compatible
    except (ValueError, TypeError):
        pass

    def _wrapped(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return fn(gold, pred, trace)

    _wrapped.__name__ = getattr(fn, "__name__", "custom_fitness_metric")
    return _wrapped


def resolve_fitness_metric(name: str,
                           custom_metrics: Dict[str, Any] = None) -> Callable:
    """Resolve a fitness metric name to a callable.

    Built-in names
    --------------
    ``"jiuwen"``  — stop-word-filtered weighted F1 (0.7 recall + 0.3 precision).
                    General-purpose; works for any skill domain. Default.
    ``"hermes"``  — word-bag overlap with 0.3 floor, matching original Hermes.

    Custom names
    ------------
    Looked up in *custom_metrics* dict first, then imported as a dotted module
    path (e.g. ``"mypackage.metrics.my_metric_fn"``).

    Raises ValueError if name cannot be resolved.
    """
    custom_metrics = custom_metrics or {}

    if name == "jiuwen":
        return jiuwen_fitness_metric
    if name == "hermes":
        return hermes_fitness_metric

    # Check custom_metrics dict
    if name in custom_metrics:
        fn = custom_metrics[name]
        if callable(fn):
            return _ensure_gepa_signature(fn)
        raise ValueError(f"custom_metrics['{name}'] is not callable: {type(fn)}")

    # Try dotted import path: "package.module.function"
    if "." in name:
        module_path, _, fn_name = name.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            fn = getattr(module, fn_name)
            if callable(fn):
                return _ensure_gepa_signature(fn)
            raise ValueError(f"'{name}' resolved but is not callable")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot import fitness metric '{name}': {e}") from e

    raise ValueError(
        f"Unknown fitness metric '{name}'. "
        f"Built-ins: 'jiuwen', 'hermes'. "
        f"For custom metrics pass a dotted import path or add to custom_fitness_metrics config."
    )
