# coding: utf-8
"""Generate ASCII charts (terminal) and a matplotlib PNG after all training."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def step(
    baseline_score: float,
    *,
    scores_no_ts:  Optional[List[float]] = None,
    scores_no_ts_multi: Optional[List[float]] = None,
    scores_l2_l3:  Optional[List[float]] = None,
    scores_l2:     Optional[List[float]] = None,
    scores_l3:     Optional[List[float]] = None,
    output_dir: Path,
    scenario_name: str = "",
    n_runs: int = 1,
) -> None:
    """Print ASCII charts to the terminal, then save a matplotlib PNG.

    ASCII charts are always shown (no extra dependencies).
    The PNG is skipped silently if matplotlib is not installed.
    """
    any_mode_ran = any(
        s for s in [scores_no_ts, scores_l2_l3, scores_l2, scores_l3] if s
    )
    if not any_mode_ran:
        return

    # ── ASCII charts (terminal) ───────────────────────────────────────────
    from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.ascii_plotter import (
        print_ascii_charts,
    )
    print_ascii_charts(
        baseline_score=baseline_score,
        scores_no_ts=scores_no_ts,
        scores_no_ts_multi=scores_no_ts_multi,
        scores_l2_l3=scores_l2_l3,
        scores_l2=scores_l2,
        scores_l3=scores_l3,
        n_runs=n_runs,
    )

    # ── Matplotlib PNG (file) ─────────────────────────────────────────────
    try:
        from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.plotter import (
            plot_results,
        )
        path = plot_results(
            baseline_score=baseline_score,
            scores_no_ts=scores_no_ts,
            scores_l2_l3=scores_l2_l3,
            scores_l2=scores_l2,
            scores_l3=scores_l3,
            output_dir=output_dir,
            scenario_name=scenario_name,
            n_runs=n_runs,
        )
        print(f"\n  PNG saved → {path}")
    except ImportError:
        pass   # matplotlib not installed — ASCII-only is fine
    except Exception as exc:
        print(f"  [plots] PNG rendering failed: {exc}")
