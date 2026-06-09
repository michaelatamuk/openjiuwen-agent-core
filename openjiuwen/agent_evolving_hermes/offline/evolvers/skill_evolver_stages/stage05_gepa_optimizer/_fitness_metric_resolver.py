from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

from ._fitness_metric_bag_of_words import fitness_metric as fitness_metric_f1
from ._fitness_metric_custom import custom_fitness_gepa_signature
from ._fitness_metric_f1 import fitness_metric as fitness_metric_bag_of_words


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
        return fitness_metric_f1
    if name == "hermes":
        return fitness_metric_bag_of_words

    # Check custom_metrics dict
    if name in custom_metrics:
        fn = custom_metrics[name]
        if callable(fn):
            return custom_fitness_gepa_signature(fn)
        raise ValueError(f"custom_metrics['{name}'] is not callable: {type(fn)}")

    # Try dotted import path: "package.module.function"
    if "." in name:
        module_path, _, fn_name = name.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            fn = getattr(module, fn_name)
            if callable(fn):
                return custom_fitness_gepa_signature(fn)
            raise ValueError(f"'{name}' resolved but is not callable")
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot import fitness metric '{name}': {e}") from e

    raise ValueError(f"Unknown fitness metric '{name}'. "
                     f"Built-ins: 'jiuwen', 'hermes'. "
                     f"For custom metrics pass a dotted import path or add to custom_fitness_metrics config.")
