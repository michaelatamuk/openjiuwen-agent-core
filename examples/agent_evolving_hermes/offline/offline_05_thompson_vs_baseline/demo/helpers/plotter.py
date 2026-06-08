# coding: utf-8
"""Matplotlib-based result plots for the Thompson Sampling demo.

Saves a single PNG to ``output_dir / "comparison.png"``.
All rendering is done with the non-interactive Agg backend so
the script works in headless / server environments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

# Use Agg before any other matplotlib import to avoid display errors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.stats import (
    bootstrap_ci_diff,
    mean,
    std,
)
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.trainings.results import (
    run_key_label,
    run_key_mode,
)

# ── Colour palette ────────────────────────────────────────────────────────────
_COLORS = {
    "Pre-train":    "#9E9E9E",
    "Pre-train-M":  "#607D8B",
    "GEPA-Holistic": "#2196F3",
    "GEPA-Rubrics":  "#00BCD4",
    "GEPA-Focused": "#FF9800",
    "GEPA-Gated":   "#4CAF50",
    "GEPA-Full":    "#E91E63",
}
_CYCLE_COLORS = ["#9C27B0", "#FF5722", "#795548", "#3F51B5", "#009688"]
_DEFAULT_COLOR = "#607D8B"
_color_cache: Dict[str, str] = {}


def _color(label: str) -> str:
    """Return a hex colour for *label*.

    Falls back to base-mode label (part before " (metric)" suffix), then
    cycles through a fixed palette for truly unknown labels.
    """
    c = _COLORS.get(label)
    if not c:
        base = label.split(" (")[0]
        c = _COLORS.get(base)
    if not c:
        if label not in _color_cache:
            idx = len(_color_cache) % len(_CYCLE_COLORS)
            _color_cache[label] = _CYCLE_COLORS[idx]
        c = _color_cache[label]
    return c or _DEFAULT_COLOR


def plot_results(
    baseline_score_holistic: float,
    baseline_score_rubrics: Optional[float],
    scores: Dict[str, List[float]],
    output_dir: Path,
    scenario_name: str = "",
    n_runs: int = 1,
) -> Path:
    """Render comparison plots and save to *output_dir/comparison.png*.

    Parameters
    ----------
    scores:
        Dict keyed by run key (e.g. ``"gepa_plain_holistic"`` or
        ``"gepa_plain_holistic__jiuwen"``).

    Returns the path of the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "comparison.png"

    # ── Assemble mode data in insertion order ─────────────────────────────
    mode_data: list[tuple[str, List[float]]] = [
        (run_key_label(k), v) for k, v in scores.items() if v
    ]
    if not mode_data:
        return save_path

    multi = n_runs > 1

    # Find reference GEPA-Uniform entry for CI panel
    uniform_entry: Optional[tuple[str, List[float]]] = next(
        ((run_key_label(k), v) for k, v in scores.items()
         if run_key_mode(k) == "gepa_plain_holistic" and v),
        None,
    )
    show_ci = multi and uniform_entry is not None and len(mode_data) > 1

    n_panels = 1 + (1 if multi else 0) + (1 if show_ci else 0)
    fig_h = 4 + 3.5 * (n_panels - 1)
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, fig_h))
    if n_panels == 1:
        axes = [axes]

    title_suffix = f" — {scenario_name}" if scenario_name else ""
    fig.suptitle(f"GEPA + Thompson Sampling{title_suffix}", fontsize=13, fontweight="bold")

    panel = 0

    # ════════════════════════════════════════════════════════════════════════
    # Panel 1: Score comparison bar chart
    # ════════════════════════════════════════════════════════════════════════
    ax = axes[panel]; panel += 1

    all_labels = ["Pre-train"] + [lbl for lbl, _ in mode_data]
    all_means  = [baseline_score_holistic] + [mean(sc) for _, sc in mode_data]
    all_stds   = [0.0] + ([std(sc) for _, sc in mode_data] if multi else [0.0] * len(mode_data))
    colors     = [_color(lbl) for lbl in all_labels]

    x = range(len(all_labels))
    bars = ax.bar(x, all_means, color=colors, alpha=0.85, zorder=3, width=0.55)

    if multi:
        ax.errorbar(
            x, all_means, yerr=all_stds,
            fmt="none", color="#333333", capsize=5, capthick=1.5,
            linewidth=1.5, zorder=4,
        )

    for bar, val, s in zip(bars, all_means, all_stds):
        lbl = f"{val:.3f}"
        if multi and s > 0:
            lbl += f"\n±{s:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (s if multi else 0) + 0.012,
            lbl, ha="center", va="bottom", fontsize=8.5,
        )

    ax.axhline(baseline_score_holistic, color=_color("Pre-train"), linestyle="--",
               linewidth=1.2, alpha=0.6, zorder=2, label="Pre-train (single)")
    if baseline_score_rubrics is not None:
        ax.axhline(baseline_score_rubrics, color=_color("Pre-train-M"), linestyle=":",
                   linewidth=1.2, alpha=0.6, zorder=2, label="Pre-train (multi)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_labels, fontsize=9, rotation=15 if len(all_labels) > 6 else 0,
                       ha="right" if len(all_labels) > 6 else "center")
    ax.set_ylabel("Holdout score", fontsize=10)
    ax.set_ylim(0, min(1.05, max(all_means) + (max(all_stds) if multi else 0) + 0.15))
    title = f"Holdout score by mode  (n={n_runs} runs per mode)" if multi else "Holdout score by mode"
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.35, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ════════════════════════════════════════════════════════════════════════
    # Panel 2: Run-by-run line chart  (multi-run only)
    # ════════════════════════════════════════════════════════════════════════
    if multi:
        ax = axes[panel]; panel += 1

        run_xs = list(range(1, n_runs + 1))
        ax.axhline(baseline_score_holistic, color=_color("Pre-train"), linestyle="--",
                   linewidth=1.2, alpha=0.5, label="Pre-train (single)", zorder=2)
        if baseline_score_rubrics is not None:
            ax.axhline(baseline_score_rubrics, color=_color("Pre-train-M"), linestyle=":",
                       linewidth=1.2, alpha=0.5, label="Pre-train (multi)", zorder=2)

        for lbl, sc in mode_data:
            ax.plot(run_xs, sc, marker="o", linewidth=1.8,
                    markersize=6, color=_color(lbl), label=lbl, zorder=3)
            m_val = mean(sc)
            s_val = std(sc)
            ax.axhspan(m_val - s_val, m_val + s_val, color=_color(lbl), alpha=0.07, zorder=1)

        ax.set_xlabel("Run", fontsize=10)
        ax.set_ylabel("Holdout score", fontsize=10)
        ax.set_xticks(run_xs)
        ax.set_title("Score per independent run  (shading = ±1 std)", fontsize=11)
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.35, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ════════════════════════════════════════════════════════════════════════
    # Panel 3: Bootstrap CI forest plot vs GEPA-Uniform
    # ════════════════════════════════════════════════════════════════════════
    if show_ci:
        ax = axes[panel]; panel += 1
        ref_label, ref_scores = uniform_entry

        ci_rows: list[tuple[str, float, float, float]] = []
        for lbl, sc in mode_data:
            if lbl == ref_label:
                continue
            d_mean, lo, hi = bootstrap_ci_diff(ref_scores, sc)
            ci_rows.append((lbl, d_mean, lo, hi))

        y_pos = list(range(len(ci_rows)))
        ax.axvline(0, color="#555555", linewidth=1.2, linestyle="--", zorder=2)

        for i, (lbl, d_mean, lo, hi) in enumerate(ci_rows):
            if lo > 0:
                c = "#2E7D32"
                verdict = "★ reliable improvement"
            elif hi < 0:
                c = "#C62828"
                verdict = "✗ reliable regression"
            else:
                c = "#616161"
                verdict = "~ inconclusive"

            ax.plot([lo, hi], [i, i], color=c, linewidth=3.5, solid_capstyle="round", zorder=3)
            ax.scatter([d_mean], [i], color=c, s=60, zorder=4, linewidths=0)
            ax.text(hi + 0.005, i, f"  {verdict}", va="center", fontsize=8.5, color=c)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([lbl for lbl, *_ in ci_rows], fontsize=10)
        ax.set_xlabel(f"Δ holdout score vs {ref_label}", fontsize=10)
        ax.set_title(f"Bootstrap 95% CI  (Δ vs {ref_label},  n={n_runs} runs)", fontsize=11)
        if ci_rows:
            x_min = min(lo for _, _, lo, _ in ci_rows) - 0.06
            x_max = max(hi for _, _, _, hi in ci_rows) + 0.18
            ax.set_xlim(x_min, x_max)
        ax.grid(axis="x", alpha=0.35, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
