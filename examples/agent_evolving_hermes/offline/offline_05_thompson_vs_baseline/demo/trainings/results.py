from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path


class DemoTrainingsResults:
    """Aggregated results from all training passes.

    When ``n_runs == 1`` the ``scores_*`` lists have exactly one element
    and the table shows a single value (identical to the old behaviour).
    When ``n_runs > 1`` they contain one score per independent run so
    that mean ± std and bootstrap confidence intervals can be computed.

    The ``metrics_*`` attributes hold the *last* run's full metrics dict
    (used for acceptance status and config rows in the comparison table).
    """

    def __init__(
        self,
        runs: List[Tuple[str, Path]],
        scores_gepa_uniform:       List[float],
        scores_gepa_full:       List[float],
        scores_gepa_focused:          List[float],
        scores_gepa_gated:          List[float],
        scores_gepa_rubric: List[float],
        metrics_gepa_uniform:       Optional[dict],
        metrics_gepa_full:       Optional[dict],
        metrics_gepa_focused:          Optional[dict],
        metrics_gepa_gated:          Optional[dict],
        metrics_gepa_rubric: Optional[dict],
    ) -> None:
        self.runs               = runs
        self.scores_gepa_uniform       = scores_gepa_uniform
        self.scores_gepa_full       = scores_gepa_full
        self.scores_gepa_focused          = scores_gepa_focused
        self.scores_gepa_gated          = scores_gepa_gated
        self.scores_gepa_rubric = scores_gepa_rubric
        self.metrics_gepa_uniform       = metrics_gepa_uniform
        self.metrics_gepa_full       = metrics_gepa_full
        self.metrics_gepa_focused          = metrics_gepa_focused
        self.metrics_gepa_gated          = metrics_gepa_gated
        self.metrics_gepa_rubric = metrics_gepa_rubric
