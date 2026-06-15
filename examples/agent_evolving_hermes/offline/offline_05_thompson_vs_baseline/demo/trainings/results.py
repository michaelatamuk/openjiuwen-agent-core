from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Canonical display label for each mode identifier.
_MODE_LABELS: Dict[str, str] = {
    "gepa_plain_holistic":              "GEPA-Plain-Holistic",
    "gepa_plain_rubrics":               "GEPA-Plain-Rubrics",
    "gepa_plain_graph":                 "GEPA-Plain-Graph",
    "gepa_plain_checklist":             "GEPA-Plain-Checklist",
    "gepa_plain_instruction_following": "GEPA-Plain-IF",
    "gepa_plain_consistency":           "GEPA-Plain-Consistency",
    "gepa_plain_comparative":           "GEPA-Plain-Comparative",
    "gepa_scoring_matrix":              "GEPA-ScoringMatrix",
    "gepa_focused_on_difficulty":       "GEPA-Focused",
    "gepa_gated":                       "GEPA-Gated",
    "gepa_full":                        "GEPA-Full",
}


def run_key_label(run_key: str) -> str:
    """Convert a run key to a human-readable label.

    Single-metric run key  ``"gepa_plain_holistic"``                  → ``"GEPA-Plain-Holistic"``
    Multi-metric run key   ``"gepa_plain_holistic__bag_of_words"``    → ``"GEPA-Plain-Holistic (bag_of_words)"``
    """
    parts = run_key.split("__", 1)
    mode_label = _MODE_LABELS.get(parts[0], parts[0])
    if len(parts) > 1:
        return f"{mode_label} ({parts[1]})"
    return mode_label


def run_key_mode(run_key: str) -> str:
    """Return the mode portion of a run key (before the first ``__``)."""
    return run_key.split("__", 1)[0]


class DemoTrainingsResults:
    """Aggregated results from all training passes.

    Replaces the fixed per-mode fields with dict-keyed storage so that any
    combination of modes and fitness metrics can be stored without schema
    changes.

    Keys
    ----
    When a single fitness metric is used the key is just the mode name
    (``"gepa_plain_holistic"``), preserving backward-compatible output directory
    names.  When multiple metrics are configured the key is
    ``"<mode>__<metric>"`` (e.g. ``"gepa_plain_holistic__jiuwen"``).

    Attributes
    ----------
    runs : List[Tuple[str, Path]]
        ``(run_key, output_dir)`` for every combination that produced at
        least one score.  Used by :func:`demo.Demo._print_skill_diff` and
        step_09 to print output-file paths.
    scores : Dict[str, List[float]]
        ``run_key → [score_run_1, score_run_2, …]``.
        ``n_runs==1`` lists contain exactly one element.
    metrics : Dict[str, Optional[dict]]
        ``run_key → last_run_metrics_dict`` (the dict returned by
        :func:`evolve_single_skill`).  ``None`` when the mode did not run.
    """

    def __init__(
        self,
        runs: List[Tuple[str, Path]],
        scores: Dict[str, List[float]],
        metrics: Dict[str, Optional[dict]],
    ) -> None:
        self.runs = runs
        self.scores = scores
        self.metrics = metrics
