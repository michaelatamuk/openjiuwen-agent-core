# coding: utf-8
"""Generate ASCII charts (terminal) and a matplotlib PNG after all training."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def run_step(baseline_score_holistic: float,
             baseline_score_rubrics: float,
             *,
             scores: Dict[str, List[float]],
             output_dir: Path,
             scenario_name: str = "",
             n_runs: int = 1,
             console=None) -> None:
    """Print ASCII charts to the terminal, then save a matplotlib PNG.

    Parameters
    ----------
    scores:
        Dict keyed by run key (e.g. ``"gepa_plain_holistic"`` or
        ``"gepa_plain_holistic__jiuwen"``).  Passed directly to the plotters.

    ASCII charts are always shown (no extra dependencies).
    The PNG is skipped silently if matplotlib is not installed.
    """
    console.print(f"\n[bold cyan]*** Demo Step 08: Plot Results Started ***[/bold cyan]")

    if not scores:
        console.print(f"*** Demo Step 08: Plot Results Finished (nothing has run) ***")
        return

    # ── ASCII charts (terminal) ───────────────────────────────────────────
    from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.ascii_plotter import (
        print_ascii_charts,
    )
    print_ascii_charts(
        baseline_score_holistic=baseline_score_holistic,
        baseline_score_rubrics=baseline_score_rubrics,
        scores=scores,
        n_runs=n_runs,
        console=console,
    )

    # ── Matplotlib PNG (file) ─────────────────────────────────────────────
    try:
        from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.plotter import (
            plot_results,
        )
        path = plot_results(
            baseline_score_holistic=baseline_score_holistic,
            baseline_score_rubrics=baseline_score_rubrics,
            scores=scores,
            output_dir=output_dir,
            scenario_name=scenario_name,
            n_runs=n_runs,
        )
        console.print(f"\n  PNG saved → {path}")
    except ImportError:
        pass   # matplotlib not installed — ASCII-only is fine
    except Exception as exc:
        console.print(f"  [plots] PNG rendering failed: {exc}")

    console.print(f"[bold cyan]*** Demo Step 08: Plot Results Finished ***[/bold cyan]")
