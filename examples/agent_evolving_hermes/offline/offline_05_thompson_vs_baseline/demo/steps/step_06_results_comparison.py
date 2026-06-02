
from __future__ import annotations

from typing import List, Optional

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.stats import (
    bootstrap_ci_diff,
    mean,
    std,
)


def step(
    baseline_score_single: float,
    baseline_score_multi: Optional[float],
    *,
    scores_no_ts:        Optional[List[float]] = None,
    scores_l2_l3:        Optional[List[float]] = None,
    scores_l2:           Optional[List[float]] = None,
    scores_l3:           Optional[List[float]] = None,
    scores_no_ts_multi:  Optional[List[float]] = None,
    metrics_no_ts:        Optional[dict] = None,
    metrics_l2_l3:        Optional[dict] = None,
    metrics_l2:           Optional[dict] = None,
    metrics_l3:           Optional[dict] = None,
    metrics_no_ts_multi:  Optional[dict] = None,
    ts_batch_size:        int = 4,
) -> None:
    """Print comparison for whichever training modes ran.

    When ``n_runs == 1`` the table shows single values (original
    behaviour).  When ``n_runs > 1`` it shows mean ± std, a per-run
    detail table (learning-curve proxy), and bootstrap 95% CIs.
    """
    # ── Collect present modes ──────────────────────────────────────────────
    mode_data: list[tuple[str, list[float], Optional[dict]]] = []
    if scores_no_ts:       mode_data.append(("No-TS",       scores_no_ts,       metrics_no_ts))
    if scores_no_ts_multi: mode_data.append(("No-TS-Multi", scores_no_ts_multi, metrics_no_ts_multi))
    if scores_l2:          mode_data.append(("L2 only",     scores_l2,          metrics_l2))
    if scores_l3:          mode_data.append(("L3 only",     scores_l3,          metrics_l3))
    if scores_l2_l3:       mode_data.append(("L2+L3",       scores_l2_l3,       metrics_l2_l3))

    ran_labels = [label for label, _, _ in mode_data]
    pre_labels = ["Pre-train Single", "Pre-train Multi"] if baseline_score_multi is not None else ["Pre-train Single"]
    all_labels = pre_labels + (ran_labels if ran_labels else ["(pre-training only)"])
    _banner("COMPARISON — " + "  ·  ".join(all_labels))

    if not mode_data:
        print(f"\n  Pre-training single holdout score: {baseline_score_single:.4f}  (no training modes ran)")
        if baseline_score_multi is not None:
            print(f"  Pre-training multi holdout score:  {baseline_score_multi:.4f}  (no training modes ran)")
        return

    n_runs = len(mode_data[0][1])  # all modes ran the same number of times
    multi = n_runs > 1

    # ── Column widths ──────────────────────────────────────────────────────
    W = 13 if multi else 11

    # Build cols: pre-train baseline columns + one column per training mode.
    # Pre-M is only included when a multi-objective baseline was evaluated.
    cols: list[tuple[str, Optional[list[float]], Optional[dict], float]] = [
        ("Pre-S", None, None, baseline_score_single),
    ]
    if baseline_score_multi is not None:
        cols.append(("Pre-M", None, None, baseline_score_multi))
    cols += [(l, s, m, 0.0) for l, s, m in mode_data]

    # Header / divider
    header  = f"  {'':32s}"
    divider = f"  {'─' * 32}"
    for label, *_ in cols:
        header  += f"  {label:>{W}}"
        divider += f"  {'─' * W}"
    print(header)
    print(divider)

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
    print(score_row)

    delta_row = f"  {'Δ over baseline':32s}"
    for label, scores, m, base in cols:
        if scores is None:
            delta_row += f"  {'—':>{W}}"
        else:
            # Use metrics-embedded baseline when available; otherwise
            # use multi baseline for multi-scored modes, single for the rest.
            mode_baseline = m.get("baseline_score") if m and "baseline_score" in m else (
                (baseline_score_multi if baseline_score_multi is not None else baseline_score_single)
                if "Multi" in label else baseline_score_single
            )
            d = mean(scores) - mode_baseline
            delta_row += f"  {('+' if d >= 0 else '') + f'{d:.4f}':>{W}}"
    print(delta_row)

    # ── Accepted (last run) ───────────────────────────────────────────────
    acc_row = f"  {'Accepted (last run)' if multi else 'Accepted':32s}"
    for _, scores, m, _ in cols:
        if scores is None:
            acc_row += f"  {'—':>{W}}"
        else:
            acc_row += f"  {'✓ yes' if m and m.get('accepted') else '✗ no':>{W}}"
    print(acc_row)

    # ── Config rows ───────────────────────────────────────────────────────
    _GATE = {
        "No-TS":       "threshold",
        "L2+L3":       "TS gate",
        "L2 only":     "threshold",
        "L3 only":     "TS gate",
        "No-TS-Multi": "threshold",
    }
    _SEL = {
        "No-TS":       "all train",
        "L2+L3":       f"top {ts_batch_size} (TS)",
        "L2 only":     f"top {ts_batch_size} (TS)",
        "L3 only":     "all train",
        "No-TS-Multi": "all train",
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
    print(gate_row)
    print(sel_row)

    # ── Winner ────────────────────────────────────────────────────────────
    # Prefer accepted modes; fall back to best score with a "(not accepted)" note.
    accepted_modes = [(label, mean(scores)) for label, scores, m in mode_data
                      if m and m.get("accepted")]
    all_modes_scored = [(label, mean(scores)) for label, scores, _ in mode_data]

    if accepted_modes:
        best_score = max(s for _, s in accepted_modes)
        ties = [k for k, s in accepted_modes if s == best_score]
        winner_str = (" = ".join(ties) + "  (tie)") if len(ties) > 1 else ties[0]
        print(f"\n  ▶  Best accepted: {winner_str}  (mean holdout {best_score:.4f})")
    else:
        best_score = max(s for _, s in all_modes_scored)
        ties = [k for k, s in all_modes_scored if s == best_score]
        winner_str = (" = ".join(ties) + "  (tie)") if len(ties) > 1 else ties[0]
        print(f"\n  ▶  Best score: {winner_str}  (mean holdout {best_score:.4f})  — not accepted")

    if not multi:
        return

    # ══════════════════════════════════════════════════════════════════════
    # MULTI-RUN SECTION
    # ══════════════════════════════════════════════════════════════════════

    # ── Per-run table (learning-curve proxy) ──────────────────────────────
    print(f"\n  Run-by-run results ({n_runs} independent runs):")
    header2, divider2 = f"  {'':22s}", f"  {'─' * 22}"
    for label, _, _, _ in cols:
        header2 += f"  {label:>{W}}"
        divider2 += f"  {'─' * W}"
    print(header2)
    print(divider2)

    for i in range(n_runs):
        row = f"  {f'Run {i + 1}':22s}"
        for _, scores, _, base in cols:
            # Use base if it's a pre-train column, otherwise the specific run score
            val = base if scores is None else scores[i]
            row += f"  {val:>{W}.4f}"
        print(row)

    print(f"  {'─' * 22}" + f"  {'─' * W}" * len(cols))
    mean_row, std_row = f"  {'Mean':22s}", f"  {'Std dev':22s}"
    for _, scores, _, base in cols:
        if scores is None:
            mean_row += f"  {base:>{W}.4f}"
            std_row += f"  {'—':>{W}}"
        else:
            mean_row += f"  {mean(scores):>{W}.4f}"
            std_row += f"  {std(scores):>{W}.4f}"
    print(mean_row)
    print(std_row)

    # ── Bootstrap CI vs No-TS ─────────────────────────────────────────────
    if scores_no_ts and len(mode_data) > 1:
        print(f"\n  Bootstrap 95% CI vs No-TS ({n_runs} runs, paired):")
        for label, scores, _ in mode_data:
            if label == "No-TS":
                continue
            d_mean, lo, hi = bootstrap_ci_diff(scores_no_ts, scores)
            sign = "+" if d_mean >= 0 else ""
            ci_str = f"[{'+' if lo >= 0 else ''}{lo:.4f}, {'+' if hi >= 0 else ''}{hi:.4f}]"
            # Significant if CI entirely above 0 (better) or below 0 (worse)
            if lo > 0:
                verdict = "★ reliably better"
            elif hi < 0:
                verdict = "✗ reliably worse"
            else:
                verdict = "~ inconclusive"
            print(f"    {label:<10}  Δ = {sign}{d_mean:.4f}  CI {ci_str}  {verdict}")
