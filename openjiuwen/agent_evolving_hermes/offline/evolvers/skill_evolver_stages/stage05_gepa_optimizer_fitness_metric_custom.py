from __future__ import annotations

import inspect
from typing import Callable


def custom_fitness_gepa_signature(fn: Callable) -> Callable:
    """Wrap *fn* so it always accepts exactly 5 positional arguments.

    GEPA inspects the metric signature and requires ``(gold, pred, trace,
    pred_name, pred_trace)``.  Built-in metrics already carry that signature;
    this wrapper ensures custom / imported metrics are compatible too.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
        positional_params = [p for p in params
                             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(positional_params) >= 5:
            return fn  # already compatible
    except (ValueError, TypeError):
        pass

    def _wrapped(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return fn(gold, pred, trace)

    _wrapped.__name__ = getattr(fn, "__name__", "custom_fitness_metric")
    return _wrapped
