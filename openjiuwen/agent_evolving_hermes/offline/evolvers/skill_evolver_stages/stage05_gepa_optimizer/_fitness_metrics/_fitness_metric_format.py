from __future__ import annotations

import re
from typing import Set

import dspy


def _detect_format_markers(text: str) -> Set[str]:
    """Detect structural format markers present in *text*.

    Returns a frozenset of marker names found.  Checked markers:

    * ``bullet_list``      — lines starting with ``-``, ``*``, or ``•``
    * ``numbered_list``    — lines starting with ``1.``, ``2)`` etc.
    * ``code_fence``       — triple-backtick blocks
    * ``markdown_headers`` — ``#`` / ``##`` headings
    * ``json``             — outermost ``{…}`` or ``[…]`` that parses as JSON
    * ``table``            — pipe-separated columns (``| … | … |``)
    * ``blockquote``       — lines starting with ``>``
    """
    markers: Set[str] = set()

    if re.search(r"^[ \t]*[\-\*\u2022][ \t]+\S", text, re.MULTILINE):
        markers.add("bullet_list")

    if re.search(r"^\d+[\.\)][ \t]+\S", text, re.MULTILINE):
        markers.add("numbered_list")

    if "```" in text:
        markers.add("code_fence")

    if re.search(r"^#{1,6}[ \t]+\S", text, re.MULTILINE):
        markers.add("markdown_headers")

    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            import json
            json.loads(stripped)
            markers.add("json")
        except Exception:
            pass

    if re.search(r"\|[^|]+\|", text):
        markers.add("table")

    if re.search(r"^[ \t]*>[ \t]+\S", text, re.MULTILINE):
        markers.add("blockquote")

    return markers


def fitness_metric(example: dspy.Example,
                   prediction: dspy.Prediction,
                   trace=None,
                   pred_name=None,
                   pred_trace=None):
    """Format-compliance metric.

    Detects structural format markers in ``example.expected_behavior``
    (bullet lists, numbered lists, code fences, Markdown headers, JSON,
    tables, blockquotes) and scores the fraction that also appear in
    ``prediction.output``.

    ::

        score = |expected_markers ∩ output_markers| / |expected_markers|

    When the rubric carries no recognisable format markers the metric
    returns 1.0 (format is unconstrained — no penalty applies).

    Why a dedicated format metric
    ------------------------------
    ``f1`` and ``bag_of_words`` only measure word overlap.  A response
    that contains all the right keywords but omits the required numbered
    list or code block still passes those metrics.  This metric catches
    structural non-compliance directly.

    It is complementary to ``f1``/``bag_of_words`` and pairs naturally
    with the ``format_adherence`` rubric dimension in the holdout judge.

    GEPA feedback
    -------------
    When called from GEPA (``pred_name`` is not None), returns a
    ``dspy.Prediction(score, feedback)`` listing which format markers
    were missing so the reflection LM can request the agent use the
    correct output structure.  When called from MIPROv2 or the
    evaluation harness (``pred_name`` is None), returns a plain float.
    """
    if not getattr(prediction, "output", "").strip():
        if pred_name is not None:
            return dspy.Prediction(score=0.0, feedback="score=0.00; response was empty")
        return 0.0

    expected_markers = _detect_format_markers(example.expected_behavior)

    if not expected_markers:
        # No specific format required — unconstrained
        if pred_name is not None:
            return dspy.Prediction(
                score=1.0,
                feedback="score=1.00; no specific format required by rubric",
            )
        return 1.0

    output_markers = _detect_format_markers(prediction.output)
    matched = expected_markers & output_markers
    missing = expected_markers - output_markers

    score = min(1.0, max(0.0, len(matched) / len(expected_markers)))

    if pred_name is not None:
        parts = [f"score={score:.2f}"]
        if missing:
            parts.append(f"missing format markers: {', '.join(sorted(missing))}")
        else:
            parts.append("all format markers present")
        if matched:
            parts.append(f"found: {', '.join(sorted(matched))}")
        return dspy.Prediction(score=score, feedback="; ".join(parts))

    return score
