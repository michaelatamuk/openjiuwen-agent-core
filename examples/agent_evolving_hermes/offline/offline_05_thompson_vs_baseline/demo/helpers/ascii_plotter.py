# coding: utf-8
"""Terminal ASCII charts for the Thompson Sampling demo.

No dependencies beyond the stdlib.  Uses Unicode block and box-drawing
characters that render correctly in any UTF-8 terminal.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.stats import (
    bootstrap_ci_diff,
    mean,
    std,
)
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.trainings.results import (
    run_key_label,
    run_key_mode,
)

# ── Colours (ANSI escape codes, skipped on terminals that don't support them) ─
_ANSI = {
    "Pre-train":      "\033[90m",   # dark grey (back-compat)
    "Base-Holistic":  "\033[90m",   # dark grey  — holistic baseline
    "Base-Rubrics":    "\033[36m",   # cyan — rubrics baseline
    "GEPA-Holistic":   "\033[94m",   # bright blue
    "GEPA-Rubrics":    "\033[96m",   # bright cyan
    "GEPA-Focused":   "\033[93m",   # bright yellow
    "GEPA-Gated":     "\033[92m",   # bright green
    "GEPA-Full":      "\033[95m",   # bright magenta
    "reset": "\033[0m",
    "green": "\033[92m",
    "red":   "\033[91m",
    "grey":  "\033[90m",
    "bold":  "\033[1m",
}

# Cycling fallback colours for additional run keys (e.g. multi-metric variants)
_CYCLE_COLORS = ["\033[33m", "\033[35m", "\033[32m", "\033[34m", "\033[31m"]
_color_cache: Dict[str, str] = {}

BAR_CHARS = "▏▎▍▌▋▊▉█"   # eighth-block glyphs for smooth bars
_BAR_W    = 50            # total horizontal bar width in characters


def _color(label: str, text: str) -> str:
    """Return *text* wrapped in the ANSI colour for *label*.

    Looks up by full label first, then by the base mode label (i.e. the
    part before a " (metric)" suffix).  Unknown labels cycle through a
    fixed fallback palette so every run key gets a distinct colour.
    """
    c = _ANSI.get(label, "")
    if not c:
        base = label.split(" (")[0]
        c = _ANSI.get(base, "")
    if not c:
        if label not in _color_cache:
            idx = len(_color_cache) % len(_CYCLE_COLORS)
            _color_cache[label] = _CYCLE_COLORS[idx]
        c = _color_cache[label]
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

def print_score_bars(baseline_score_holistic: float,
                     mode_data: List[Tuple[str, List[float]]],
                     multi: bool = False,
                     baseline_score_rubrics: Optional[float] = None,
                     console=None) -> None:
    """Print a horizontal bar chart of holdout scores to stdout."""
    pre_items: list[tuple[str, List[float]]] = [("Base-Holistic", [baseline_score_holistic])]
    if baseline_score_rubrics is not None:
        pre_items.append(("Base-Rubrics", [baseline_score_rubrics]))
    all_items: list[tuple[str, List[float]]] = pre_items + mode_data

    max_score = max(mean(sc) for _, sc in all_items)
    label_w   = max(len(lbl) for lbl, _ in all_items)
    title     = "Holdout scores" + (" (mean ± std)" if multi else "")

    border = "─" * (label_w + _BAR_W + 22)
    console.print(f"\n  {_ANSI['bold']}{title}{_ANSI['reset']}")
    console.print(f"  ┌{border}┐")

    for lbl, scores in all_items:
        m = mean(scores)
        s = std(scores) if multi and len(scores) > 1 else 0.0
        bar = _bar(m, max_score)
        val = f"{m:.4f}"
        if multi and s > 0.0001:
            val += f" ±{s:.4f}"
        colored_bar = _color(lbl, bar)
        console.print(f"  │  {lbl:<{label_w}}  {colored_bar}  {val:<16}│")

    console.print(f"  └{border}┘")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 2: Run-by-run sparkline table
# ══════════════════════════════════════════════════════════════════════════════

def print_run_sparklines(baseline_score: float,
                         mode_data: List[Tuple[str, List[float]]],
                         n_runs: int,
                         console=None) -> None:
    """One row per mode, each cell shows that run's score as a tiny bar."""
    if n_runs < 2:
        return

    CELL_W   = 7      # chars per run cell
    label_w  = max(len(lbl) for lbl, _ in mode_data)
    all_scores = [s for _, sc in mode_data for s in sc]
    max_score  = max(all_scores) if all_scores else 1.0

    header_cells = "  ".join(f"{'Run ' + str(i):^{CELL_W}}" for i in range(1, n_runs + 1))
    border = "─" * (label_w + (CELL_W + 2) * n_runs + 6)

    console.print(f"\n  {_ANSI['bold']}Run-by-run scores{_ANSI['reset']}")
    console.print(f"  ┌{border}┐")
    console.print(f"  │  {' ' * label_w}  {header_cells}  │")
    console.print(f"  ├{'─' * (len(border))}┤")

    for lbl, scores in mode_data:
        cells = []
        for s in scores:
            bar_len = round(s / max_score * CELL_W)
            cell = "█" * bar_len + "░" * (CELL_W - bar_len)
            cells.append(f"{_color(lbl, cell)}")
        console.print(f"  │  {lbl:<{label_w}}  {'  '.join(cells)}  │")
        vals = "  ".join(f"{s:^{CELL_W}.4f}" for s in scores)
        console.print(f"  │  {' ' * label_w}  {vals}  │")

    console.print(f"  └{border}┘")
    console.print(f"  {'Pre-train baseline':>{label_w + 4}}: {baseline_score:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 3: Bootstrap CI forest plot
# ══════════════════════════════════════════════════════════════════════════════

def print_ci_forest(ref_scores: List[float],
                    ref_label: str,
                    mode_data: List[Tuple[str, List[float]]],
                    n_runs: int,
                    width: int = 44,
                    console=None) -> None:
    """Horizontal CI bars for each mode vs the reference (*ref_label*)."""
    if not ref_scores or len(mode_data) <= 1:
        return

    ci_rows: list[tuple[str, float, float, float]] = []
    for lbl, scores in mode_data:
        if lbl == ref_label:
            continue
        d_mean, lo, hi = bootstrap_ci_diff(ref_scores, scores)
        ci_rows.append((lbl, d_mean, lo, hi))

    if not ci_rows:
        return

    x_min = min(lo for _, _, lo, _ in ci_rows) - 0.03
    x_max = max(hi for _, _, _, hi in ci_rows) + 0.03

    def to_col(v: float) -> int:
        return min(width - 1, max(0, round((v - x_min) / (x_max - x_min) * (width - 1))))

    zero_col = to_col(0.0)
    label_w  = max(len(lbl) for lbl, *_ in ci_rows)

    console.print(f"\n  {_ANSI['bold']}Bootstrap 95% CI  vs {ref_label}  (n={n_runs} runs){_ANSI['reset']}")
    border = "─" * (label_w + width + 28)
    console.print(f"  ┌{border}┐")

    axis_row = [" "] * width
    axis_row[zero_col] = "│"
    console.print(f"  │  {' ' * label_w}  {''.join(axis_row)}  {'':26}│")

    for lbl, d_mean, lo, hi in ci_rows:
        lo_c = to_col(lo)
        hi_c = to_col(hi)
        mu_c = to_col(d_mean)

        if lo > 0:
            verdict_str = f"{_ANSI['green']}★ reliable improvement{_ANSI['reset']}"
            ci_color = _ANSI["green"]
        elif hi < 0:
            verdict_str = f"{_ANSI['red']}✗ reliable regression {_ANSI['reset']}"
            ci_color = _ANSI["red"]
        else:
            verdict_str = f"{_ANSI['grey']}~ inconclusive        {_ANSI['reset']}"
            ci_color = _ANSI["grey"]

        row = [" "] * width
        row[zero_col] = "│"
        for i in range(lo_c, hi_c + 1):
            row[i] = "─"
        row[lo_c] = "["
        row[hi_c] = "]"
        row[mu_c] = "●"

        ci_str   = ci_color + "".join(row) + _ANSI["reset"]
        sign     = "+" if d_mean >= 0 else ""
        delta    = f"{sign}{d_mean:.4f}"
        ci_range = f"[{'+' if lo >= 0 else ''}{lo:.4f},{'+' if hi >= 0 else ''}{hi:.4f}]"
        console.print(f"  │  {lbl:<{label_w}}  {ci_str}  {delta} {ci_range} {verdict_str}│")

    tick_row = [" "] * width
    tick_row[zero_col] = "┴"
    console.print(f"  │  {' ' * label_w}  {''.join(tick_row)}  {'':26}│")

    lbl_row = [" "] * width
    zero_lbl = "0"
    insert_at = max(0, zero_col - len(zero_lbl) // 2)
    for k, ch in enumerate(zero_lbl):
        if insert_at + k < width:
            lbl_row[insert_at + k] = ch
    console.print(f"  │  {' ' * label_w}  {''.join(lbl_row)}  {'':26}│")
    console.print(f"  └{border}┘")


# ══════════════════════════════════════════════════════════════════════════════
# Combined entry point
# ══════════════════════════════════════════════════════════════════════════════

def print_ascii_charts(baseline_score_holistic: float,
                       baseline_score_rubrics: Optional[float],
                       scores: Dict[str, List[float]],
                       n_runs: int = 1,
                       console=None) -> None:
    """Print all relevant ASCII charts to stdout.

    Parameters
    ----------
    scores:
        Dict keyed by run key (e.g. ``"gepa_plain_holistic"`` or
        ``"gepa_plain_holistic__jiuwen"``).
    """
    mode_data: list[tuple[str, List[float]]] = [
        (run_key_label(k), v) for k, v in scores.items() if v
    ]
    if not mode_data:
        return

    multi = n_runs > 1

    # Chart 1 — always
    print_score_bars(baseline_score_holistic, mode_data, multi=multi,
                     baseline_score_rubrics=baseline_score_rubrics, console=console)

    # Chart 2 — only when multiple runs
    if multi:
        sparkline_baseline = baseline_score_rubrics if baseline_score_rubrics is not None else baseline_score_holistic
        print_run_sparklines(sparkline_baseline, mode_data, n_runs, console)

    # Chart 3 — only when multiple runs AND a gepa_plain_holistic entry exists
    if multi:
        uniform_entry = next(
            ((run_key_label(k), v) for k, v in scores.items()
             if run_key_mode(k) == "gepa_plain_holistic" and v),
            None,
        )
        if uniform_entry and len(mode_data) > 1:
            ref_label, ref_scores = uniform_entry
            print_ci_forest(ref_scores, ref_label, mode_data, n_runs, console=console)
