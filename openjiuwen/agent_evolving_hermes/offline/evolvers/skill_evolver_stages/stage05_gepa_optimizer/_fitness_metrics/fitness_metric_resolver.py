from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

from ._fitness_metric_bag_of_words import fitness_metric as fitness_metric_bag_of_words
from ._fitness_metric_custom import custom_fitness_gepa_signature
from ._fitness_metric_f1 import fitness_metric as fitness_metric_f1
from ._fitness_metric_format import fitness_metric as fitness_metric_format
from ._fitness_metric_graph import fitness_metric as fitness_metric_graph
from ._fitness_metric_ner import fitness_metric as fitness_metric_ner
from ._fitness_metric_rouge_l import fitness_metric as fitness_metric_rouge_l
from ._fitness_metric_semantic import fitness_metric as fitness_metric_semantic


def resolve_fitness_metric(name: str,
                           custom_metrics: Dict[str, Any] = None) -> Callable:
    """Resolve a fitness metric name to a callable.

    Built-in names
    --------------
    ``"f1"``            — stop-word-filtered weighted F1
                            (0.7 recall + 0.3 precision).
                            General-purpose; works for any skill domain. Default.
    ``"bag_of_words"``  — word-bag overlap with 0.3 floor,
                            matching original Hermes behaviour.
    ``"graph"``         — concept-graph structural similarity.
                            Converts expected behavior and agent output to
                            concept graphs (unigram/bigram nodes + co-occurrence
                            edges) and scores their structural overlap.
                            Differentiates responses that use correct concepts
                            in the right relational context vs. scattered
                            keyword matches.
    ``"rouge_l"``       — ROUGE-L (Longest Common Subsequence F1).
                            Order-preserving token overlap; penalises responses
                            that cover required keywords but in a different order.
                            Useful for procedural / step-by-step skills.
    ``"semantic"``      — Sentence-transformer cosine similarity
                            (``all-MiniLM-L6-v2``).  Captures paraphrase and
                            synonymy that lexical metrics miss.
                            Requires: ``pip install sentence-transformers``
    ``"format"``        — Format-compliance marker detection.
                            Checks whether the response uses the same structural
                            markers as the rubric (bullet lists, numbered lists,
                            code fences, Markdown headers, JSON, tables).
    ``"ner"``           — Named-entity coverage (recall-biased F1).
                            Extracts entities via spaCy ``en_core_web_sm``;
                            falls back to capitalised-word heuristic when spaCy
                            is unavailable.
                            Requires: ``pip install spacy && python -m spacy download en_core_web_sm``

    Custom names
    ------------
    Looked up in *custom_metrics* dict first, then imported as a dotted module
    path (e.g. ``"mypackage.metrics.my_metric_fn"``).

    Raises ValueError if name cannot be resolved.
    """
    custom_metrics = custom_metrics or {}

    if name in ("f1",):
        return fitness_metric_f1
    if name in ("bag_of_words",):
        return fitness_metric_bag_of_words
    if name in ("graph",):
        return fitness_metric_graph
    if name in ("rouge_l",):
        return fitness_metric_rouge_l
    if name in ("semantic",):
        return fitness_metric_semantic
    if name in ("format",):
        return fitness_metric_format
    if name in ("ner",):
        return fitness_metric_ner

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

    raise ValueError(
        f"Unknown fitness metric '{name}'. "
        f"Built-ins: 'f1', 'bag_of_words', 'graph', 'rouge_l', 'semantic', 'format', 'ner'. "
        f"For custom metrics pass a dotted import path or add to custom_fitness_metrics config."
    )
