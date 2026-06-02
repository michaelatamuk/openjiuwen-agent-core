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
        scores_no_ts:       List[float],
        scores_l2_l3:       List[float],
        scores_l2:          List[float],
        scores_l3:          List[float],
        scores_no_ts_multi: List[float],
        metrics_no_ts:       Optional[dict],
        metrics_l2_l3:       Optional[dict],
        metrics_l2:          Optional[dict],
        metrics_l3:          Optional[dict],
        metrics_no_ts_multi: Optional[dict],
    ) -> None:
        self.runs               = runs
        self.scores_no_ts       = scores_no_ts
        self.scores_l2_l3       = scores_l2_l3
        self.scores_l2          = scores_l2
        self.scores_l3          = scores_l3
        self.scores_no_ts_multi = scores_no_ts_multi
        self.metrics_no_ts       = metrics_no_ts
        self.metrics_l2_l3       = metrics_l2_l3
        self.metrics_l2          = metrics_l2
        self.metrics_l3          = metrics_l3
        self.metrics_no_ts_multi = metrics_no_ts_multi
