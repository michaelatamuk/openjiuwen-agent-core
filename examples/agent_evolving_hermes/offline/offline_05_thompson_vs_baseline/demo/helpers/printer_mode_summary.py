from __future__ import annotations

from typing import List, Optional

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.stats import (
    mean,
    std,
)

_SEP = "─" * 48


def print_mode_summary(
    mode_label: str,
    baseline_score: Optional[float],
    scores: List[float],
    elapsed_sec: Optional[float] = None,
) -> None:
    """Print a compact per-mode summary after all runs of a mode complete.

    Only call this when len(scores) > 1.
    """
    n = len(scores)
    m = mean(scores)
    s = std(scores)
    best_i = max(range(n), key=lambda i: scores[i])

    elapsed_str = f"  ({_fmt_duration(elapsed_sec)})" if elapsed_sec is not None else ""
    print(f"\n  {mode_label} — {n} runs summary{elapsed_str}")
    print(f"  {_SEP}")
    print(f"  {'':20s}  {'Score':>9}  {'Δ baseline':>12}")
    print(f"  {_SEP}")

    # Baseline row
    bs_str = f"{baseline_score:.4f}" if baseline_score is not None else "—"
    print(f"  {'Pre-train':20s}  {bs_str:>9}  {'—':>12}")

    # One row per run
    for i, sc in enumerate(scores, start=1):
        tag = "  ◀ best" if (i - 1) == best_i else ""
        label = f"Run {i}{tag}"
        if baseline_score is not None:
            d = sc - baseline_score
            delta_str = f"{'+' if d >= 0 else ''}{d:.4f}"
        else:
            delta_str = "—"
        print(f"  {label:20s}  {sc:>9.4f}  {delta_str:>12}")

    print(f"  {_SEP}")

    # Mean ± std (spans both numeric columns)
    mean_str = f"{m:.4f} ± {s:.4f}"
    print(f"  {'Mean ± std':20s}  {mean_str:>23}")

    # Best run line
    if baseline_score is not None:
        best_d = scores[best_i] - baseline_score
        sign = "+" if best_d >= 0 else ""
        best_str = (
            f"Run {best_i + 1}   "
            f"{scores[best_i]:.4f}   "
            f"({sign}{best_d:.4f})"
        )
        print(f"  {'Best run':20s}  {best_str}")


def _fmt_duration(sec: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def print_mode_timing(mode_label: str, elapsed_sec: float) -> None:
    """Print a single timing line for a mode that ran exactly once."""
    print(f"  {mode_label} completed in {_fmt_duration(elapsed_sec)}")
