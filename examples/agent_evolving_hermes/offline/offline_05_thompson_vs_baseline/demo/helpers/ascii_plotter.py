# coding: utf-8
"""Terminal ASCII charts for the Thompson Sampling demo.

No dependencies beyond the stdlib.  Uses Unicode block and box-drawing
characters that render correctly in any UTF-8 terminal.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.stats import (
    bootstrap_ci_diff,
    mean,
    std,
)

# ── Colours (ANSI escape codes, skipped on terminals that don't support them) ─
_ANSI = {
    "Pre-train": "\033[90m",   # dark grey
    "No-TS":     "\033[94m",   # bright blue
    "L2 only":   "\033[93m",   # bright yellow
    "L3 only":   "\033[92m",   # bright green
    "L2+L3":     "\033[95m",   # bright magenta
    "reset":     "\033[0m",
    "green":     "\033[92m",
    "red":       "\033[91m",
    "grey":      "\033[90m",
    "bold":      "\033[1m",
}

BAR_CHARS = "▏▎▍▌▋▊▉█"   # eighth-block glyphs for smooth bars
_BAR_W    = 36            # total horizontal bar width in characters


def _color(label: str, text: str) -> str:
    c = _ANSI.get(label, "")
    return f"{c}{text}{_ANSI['reset']}" if c else text


def _bar(value: float, max_value: float, width: int = _BAR_W) -> str:
    """Return a smooth Unicode block bar scaled to *width* chars."""
    if max_value <= 0:
        return " " * width
    frac = min(value / max_value, 1.0)
    total_eighths = round(frac * width * 8)
    full_blocks, remainder = divmod(total_eighths, 8)
    bar = "█" * full_blocks
    if remainder and full_blocks < width:
        bar += BAR_CHARS[remainder - 1]
    return bar.ljust(width)


# ══════════════════════════════════════════════════════════════════════════════
# Chart 1: Score bar chart
# ══════════════════════════════════════════════════════════════════════════════

def print_score_bars(
    baseline_score: float,
    mode_data: List[Tuple[str, List[float]]],
    multi: bool = False,
) -> None:
    """Print a horizontal bar chart of holdout scores to stdout."""
    all_items: list[tuple[str, List[float]]] = [
        ("Pre-train", [baseline_score])
    ] + mode_data

    max_score = max(mean(sc) for _, sc in all_items)
    label_w   = max(len(lbl) for lbl, _ in all_items)
    title     = "Holdout scores" + (" (mean ± std)" if multi else "")

    border = "─" * (label_w + _BAR_W + 22)
    print(f"\n  {_ANSI['bold']}{title}{_ANSI['reset']}")
    print(f"  ┌{border}┐")

    for lbl, scores in all_items:
        m = mean(scores)
        s = std(scores) if multi and len(scores) > 1 else 0.0
        bar = _bar(m, max_score)
        val = f"{m:.4f}"
        if multi and s > 0.0001:
            val += f" ±{s:.4f}"
        colored_bar = _color(lbl, bar)
        print(f"  │  {lbl:<{label_w}}  {colored_bar}  {val:<16}│")

    print(f"  └{border}┘")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 2: Run-by-run sparkline table
# ══════════════════════════════════════════════════════════════════════════════

def print_run_sparklines(
    baseline_score: float,
    mode_data: List[Tuple[str, List[float]]],
    n_runs: int,
) -> None:
    """One row per mode, each cell shows that run's score as a tiny bar."""
    if n_runs < 2:
        return

    CELL_W   = 7      # chars per run cell
    label_w  = max(len(lbl) for lbl, _ in mode_data)
    all_scores = [s for _, sc in mode_data for s in sc]
    max_score  = max(all_scores) if all_scores else 1.0

    header_cells = "  ".join(f"{'Run ' + str(i):^{CELL_W}}" for i in range(1, n_runs + 1))
    border = "─" * (label_w + (CELL_W + 2) * n_runs + 6)

    print(f"\n  {_ANSI['bold']}Run-by-run scores{_ANSI['reset']}")
    print(f"  ┌{border}┐")
    print(f"  │  {' ' * label_w}  {header_cells}  │")
    print(f"  ├{'─' * (len(border))}┤")

    for lbl, scores in mode_data:
        cells = []
        for s in scores:
            bar_len = round(s / max_score * CELL_W)
            cell = "█" * bar_len + "░" * (CELL_W - bar_len)
            cells.append(f"{_color(lbl, cell)}")
        print(f"  │  {lbl:<{label_w}}  {'  '.join(cells)}  │")
        # numeric sub-row
        vals = "  ".join(f"{s:^{CELL_W}.4f}" for s in scores)
        print(f"  │  {' ' * label_w}  {vals}  │")

    print(f"  └{border}┘")
    print(f"  {'Pre-train baseline':>{label_w + 4}}: {baseline_score:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 3: Bootstrap CI forest plot
# ══════════════════════════════════════════════════════════════════════════════

def print_ci_forest(
    scores_no_ts: List[float],
    mode_data: List[Tuple[str, List[float]]],
    n_runs: int,
    width: int = 44,
) -> None:
    """Horizontal CI bars for each TS mode vs No-TS."""
    if not scores_no_ts or len(mode_data) <= 1:
        return

    # Build CI data
    ci_rows: list[tuple[str, float, float, float]] = []
    for lbl, scores in mode_data:
        if lbl == "No-TS":
            continue
        d_mean, lo, hi = bootstrap_ci_diff(scores_no_ts, scores)
        ci_rows.append((lbl, d_mean, lo, hi))

    if not ci_rows:
        return

    x_min = min(lo for _, _, lo, _ in ci_rows) - 0.03
    x_max = max(hi for _, _, _, hi in ci_rows) + 0.03

    def to_col(v: float) -> int:
        return min(width - 1, max(0, round((v - x_min) / (x_max - x_min) * (width - 1))))

    zero_col = to_col(0.0)
    label_w  = max(len(lbl) for lbl, *_ in ci_rows)

    print(f"\n  {_ANSI['bold']}Bootstrap 95% CI  vs No-TS  (n={n_runs} runs){_ANSI['reset']}")
    border = "─" * (label_w + width + 28)
    print(f"  ┌{border}┐")

    # Axis header row
    axis_row = [" "] * width
    axis_row[zero_col] = "│"
    axis_str = "".join(axis_row)
    print(f"  │  {' ' * label_w}  {axis_str}  {'':26}│")

    for lbl, d_mean, lo, hi in ci_rows:
        lo_c   = to_col(lo)
        hi_c   = to_col(hi)
        mu_c   = to_col(d_mean)

        if lo > 0:
            verdict_str = f"{_ANSI['green']}★ reliable improvement{_ANSI['reset']}"
            ci_color = _ANSI["green"]
        elif hi < 0:
            verdict_str = f"{_ANSI['red']}✗ reliable regression {_ANSI['reset']}"
            ci_color = _ANSI["red"]
        else:
            verdict_str = f"{_ANSI['grey']}~ inconclusive        {_ANSI['reset']}"
            ci_color = _ANSI["grey"]

        # Build CI bar
        row = [" "] * width
        row[zero_col] = "│"
        for i in range(lo_c, hi_c + 1):
            row[i] = "─"
        row[lo_c] = "["
        row[hi_c] = "]"
        # Mean marker (overwrite dash/bracket)
        row[mu_c] = "●"

        ci_str   = ci_color + "".join(row) + _ANSI["reset"]
        sign     = "+" if d_mean >= 0 else ""
        delta    = f"{sign}{d_mean:.4f}"
        ci_range = f"[{'+' if lo >= 0 else ''}{lo:.4f},{'+' if hi >= 0 else ''}{hi:.4f}]"
        print(f"  │  {lbl:<{label_w}}  {ci_str}  {delta} {ci_range} {verdict_str}│")

    # Axis tick row
    tick_row = [" "] * width
    tick_row[zero_col] = "┴"
    print(f"  │  {' ' * label_w}  {''.join(tick_row)}  {'':26}│")

    # Zero label row
    lbl_row = [" "] * width
    zero_lbl = "0"
    insert_at = max(0, zero_col - len(zero_lbl) // 2)
    for k, ch in enumerate(zero_lbl):
        if insert_at + k < width:
            lbl_row[insert_at + k] = ch
    print(f"  │  {' ' * label_w}  {''.join(lbl_row)}  {'':26}│")

    print(f"  └{border}┘")


# ══════════════════════════════════════════════════════════════════════════════
# Combined entry point
# ══════════════════════════════════════════════════════════════════════════════

def print_ascii_charts(
    baseline_score: float,
    scores_no_ts:  Optional[List[float]],
    scores_l2_l3:  Optional[List[float]],
    scores_l2:     Optional[List[float]],
    scores_l3:     Optional[List[float]],
    n_runs: int = 1,
) -> None:
    """Print all relevant ASCII charts to stdout."""
    mode_data: list[tuple[str, List[float]]] = []
    if scores_no_ts:  mode_data.append(("No-TS",   scores_no_ts))
    if scores_l2:     mode_data.append(("L2 only", scores_l2))
    if scores_l3:     mode_data.append(("L3 only", scores_l3))
    if scores_l2_l3:  mode_data.append(("L2+L3",   scores_l2_l3))

    if not mode_data:
        return

    multi = n_runs > 1

    # Chart 1 — always
    print_score_bars(baseline_score, mode_data, multi=multi)

    # Chart 2 — only when multiple runs
    if multi:
        print_run_sparklines(baseline_score, mode_data, n_runs)

    # Chart 3 — only when multiple runs AND No-TS present
    if multi and scores_no_ts:
        print_ci_forest(scores_no_ts, mode_data, n_runs)
