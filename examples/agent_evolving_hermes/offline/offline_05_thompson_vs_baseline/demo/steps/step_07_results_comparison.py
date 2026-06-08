
from __future__ import annotations

from typing import List, Optional

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.stats import (
    bootstrap_ci_diff,
    mean,
    std,
)


def run_step(baseline_score_single: float,
             baseline_score_multi: Optional[float],
             *,
             scores_gepa_uniform:        Optional[List[float]] = None,
             scores_gepa_full:        Optional[List[float]] = None,
             scores_gepa_focused:           Optional[List[float]] = None,
             scores_gepa_gated:           Optional[List[float]] = None,
             scores_gepa_rubric:  Optional[List[float]] = None,
             metrics_gepa_uniform:        Optional[dict] = None,
             metrics_gepa_full:        Optional[dict] = None,
             metrics_gepa_focused:           Optional[dict] = None,
             metrics_gepa_gated:           Optional[dict] = None,
             metrics_gepa_rubric:  Optional[dict] = None,
             ts_batch_size:        int = 4,
             console=None) -> None:
    """Print comparison for whichever training modes ran.

    When ``n_runs == 1`` the table shows single values (original
    behaviour).  When ``n_runs > 1`` it shows mean ± std, a per-run
    detail table (learning-curve proxy), and bootstrap 95% CIs.
    """

    console.print(f"\n[bold cyan]*** Demo Step 07: Compare Results Started ***[/bold cyan]")

    # ── Collect present modes ──────────────────────────────────────────────
    mode_data: list[tuple[str, list[float], Optional[dict]]] = []
    if scores_gepa_uniform:       mode_data.append(("GEPA-Uniform",       scores_gepa_uniform,       metrics_gepa_uniform))
    if scores_gepa_rubric: mode_data.append(("GEPA-Rubric", scores_gepa_rubric, metrics_gepa_rubric))
    if scores_gepa_focused:          mode_data.append(("GEPA-Focused",     scores_gepa_focused,          metrics_gepa_focused))
    if scores_gepa_gated:          mode_data.append(("GEPA-Gated",     scores_gepa_gated,          metrics_gepa_gated))
    if scores_gepa_full:       mode_data.append(("GEPA-Full",       scores_gepa_full,       metrics_gepa_full))

    ran_labels = [label for label, _, _ in mode_data]
    pre_labels = ["Base-Holistic", "Base-Rubric"] if baseline_score_multi is not None else ["Base-Holistic"]
    all_labels = pre_labels + (ran_labels if ran_labels else ["(pre-training only)"])
    _banner("COMPARISON — " + "  ·  ".join(all_labels), console)

    if not mode_data:
        console.print(f"\n  Pre-training single holdout score: {baseline_score_single:.4f}  (no training modes ran)")
        if baseline_score_multi is not None:
            console.print(f"  Pre-training multi holdout score:  {baseline_score_multi:.4f}  (no training modes ran)")
        return

    n_runs = len(mode_data[0][1])  # all modes ran the same number of times
    multi = n_runs > 1

    # ── Column widths ──────────────────────────────────────────────────────
    W = 13 if multi else 11

    # Build cols: pre-train baseline columns + one column per training mode.
    # Base-Rubric is only included when a multi-objective baseline was evaluated.
    cols: list[tuple[str, Optional[list[float]], Optional[dict], float]] = [
        ("Base-Holistic", None, None, baseline_score_single),
    ]
    if baseline_score_multi is not None:
        cols.append(("Base-Rubric", None, None, baseline_score_multi))
    cols += [(l, s, m, 0.0) for l, s, m in mode_data]

    # Header / divider
    header  = f"  {'':32s}"
    divider = f"  {'─' * 32}"
    for label, *_ in cols:
        header  += f"  {label:>{W}}"
        divider += f"  {'─' * W}"
    console.print(header)
    console.print(divider)

    # Holdout score row
    score_row = f"  {'Holdout score':32s}"
    for label, scores, m, base in cols:
        if scores is None:
            score_row += f"  {base:>{W}.4f}"  # Dynamically use the pre-train baseline
        elif multi:
            m_val = mean(scores)
            s_val = std(scores)
            score_row += f"  {f'{m_val:.4f} ±{s_val:.4f}':>{W}}"
        else:
            score_row += f"  {scores[0]:>{W}.4f}"
    console.print(score_row)

    delta_row = f"  {'Δ over baseline':32s}"
    for label, scores, m, base in cols:
        if scores is None:
            delta_row += f"  {'—':>{W}}"
        else:
            # Use metrics-embedded baseline when available; otherwise
            # use multi baseline for multi-scored modes, single for the rest.
            mode_baseline = m.get("baseline_score") if m and "baseline_score" in m else (
                (baseline_score_multi if baseline_score_multi is not None else baseline_score_single)
                if label == "GEPA-Rubric" else baseline_score_single
            )
            d = mean(scores) - mode_baseline
            delta_row += f"  {('+' if d >= 0 else '') + f'{d:.4f}':>{W}}"
    console.print(delta_row)

    # ── Accepted (last run) ───────────────────────────────────────────────
    acc_row = f"  {'Accepted (last run)' if multi else 'Accepted':32s}"
    for _, scores, m, _ in cols:
        if scores is None:
            acc_row += f"  {'—':>{W}}"
        else:
            acc_row += f"  {'✓ yes' if m and m.get('accepted') else '✗ no':>{W}}"
    console.print(acc_row)

    # ── Config rows ───────────────────────────────────────────────────────
    _GATE = {
        "GEPA-Uniform":       "threshold",
        "GEPA-Full":       "TS gate",
        "GEPA-Focused":     "threshold",
        "GEPA-Gated":     "TS gate",
        "GEPA-Rubric": "threshold",
    }
    _SEL = {
        "GEPA-Uniform":       "all train",
        "GEPA-Full":       f"top {ts_batch_size} (TS)",
        "GEPA-Focused":     f"top {ts_batch_size} (TS)",
        "GEPA-Gated":     "all train",
        "GEPA-Rubric": "all train",
    }

    gate_row = f"  {'Acceptance gate':32s}"
    sel_row = f"  {'Example selector':32s}"
    for label, scores, _, _ in cols:
        if scores is None:
            gate_row += f"  {'—':>{W}}"
            sel_row += f"  {'—':>{W}}"
        else:
            gate_row += f"  {_GATE.get(label, '?'):>{W}}"
            sel_row += f"  {_SEL.get(label, '?'):>{W}}"
    console.print(gate_row)
    console.print(sel_row)

    # ── Winner ────────────────────────────────────────────────────────────
    # Prefer accepted modes; fall back to best score with a "(not accepted)" note.
    accepted_modes = [(label, mean(scores)) for label, scores, m in mode_data
                      if m and m.get("accepted")]
    all_modes_scored = [(label, mean(scores)) for label, scores, _ in mode_data]

    if accepted_modes:
        best_score = max(s for _, s in accepted_modes)
        ties = [k for k, s in accepted_modes if s == best_score]
        winner_str = (" = ".join(ties) + "  (tie)") if len(ties) > 1 else ties[0]
        console.print(f"\n  ▶  Best accepted: {winner_str}  (mean holdout {best_score:.4f})")
    else:
        best_score = max(s for _, s in all_modes_scored)
        ties = [k for k, s in all_modes_scored if s == best_score]
        winner_str = (" = ".join(ties) + "  (tie)") if len(ties) > 1 else ties[0]
        console.print(f"\n  ▶  Best score: {winner_str}  (mean holdout {best_score:.4f})  — not accepted")

    if not multi:
        console.print(f"*** Demo Step 07: Compare Results Finished (not multi) ***")
        return

    # ══════════════════════════════════════════════════════════════════════
    # MULTI-RUN SECTION
    # ══════════════════════════════════════════════════════════════════════

    # ── Per-run table (learning-curve proxy) ──────────────────────────────
    console.print(f"\n  Run-by-run results ({n_runs} independent runs):")
    header2, divider2 = f"  {'':22s}", f"  {'─' * 22}"
    for label, _, _, _ in cols:
        header2 += f"  {label:>{W}}"
        divider2 += f"  {'─' * W}"
    console.print(header2)
    console.print(divider2)

    for i in range(n_runs):
        row = f"  {f'Run {i + 1}':22s}"
        for _, scores, _, base in cols:
            # Use base if it's a pre-train column, otherwise the specific run score
            val = base if scores is None else scores[i]
            row += f"  {val:>{W}.4f}"
        console.print(row)

    console.print(f"  {'─' * 22}" + f"  {'─' * W}" * len(cols))
    mean_row, std_row = f"  {'Mean':22s}", f"  {'Std dev':22s}"
    for _, scores, _, base in cols:
        if scores is None:
            mean_row += f"  {base:>{W}.4f}"
            std_row += f"  {'—':>{W}}"
        else:
            mean_row += f"  {mean(scores):>{W}.4f}"
            std_row += f"  {std(scores):>{W}.4f}"
    console.print(mean_row)
    console.print(std_row)

    # ── Bootstrap CI vs GEPA-Uniform ─────────────────────────────────────────────
    if scores_gepa_uniform and len(mode_data) > 1:
        console.print(f"\n  Bootstrap 95% CI vs GEPA-Uniform ({n_runs} runs, paired):")
        for label, scores, _ in mode_data:
            if label == "GEPA-Uniform":
                continue
            d_mean, lo, hi = bootstrap_ci_diff(scores_gepa_uniform, scores)
            sign = "+" if d_mean >= 0 else ""
            ci_str = f"[{'+' if lo >= 0 else ''}{lo:.4f}, {'+' if hi >= 0 else ''}{hi:.4f}]"
            # Significant if CI entirely above 0 (better) or below 0 (worse)
            if lo > 0:
                verdict = "★ reliably better"
            elif hi < 0:
                verdict = "✗ reliably worse"
            else:
                verdict = "~ inconclusive"
            console.print(f"    {label:<10}  Δ = {sign}{d_mean:.4f}  CI {ci_str}  {verdict}")

    console.print(f"[bold cyan]*** Demo Step 07: Compare Results Finished ***[/bold cyan]")
