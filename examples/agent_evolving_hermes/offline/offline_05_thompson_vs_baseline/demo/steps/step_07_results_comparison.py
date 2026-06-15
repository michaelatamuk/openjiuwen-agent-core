
from __future__ import annotations

from typing import Dict, List, Optional

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.stats import (
    bootstrap_ci_diff,
    mean,
    std,
)
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.trainings.results import (
    run_key_label,
    run_key_mode,
)

# Per-mode acceptance gate type
_GATE = {
    "gepa_plain_holistic":              "threshold",
    "gepa_plain_rubrics":               "threshold",
    "gepa_plain_graph":                 "threshold",
    "gepa_plain_checklist":             "threshold",
    "gepa_plain_instruction_following": "threshold",
    "gepa_plain_consistency":           "threshold",
    "gepa_plain_comparative":           "threshold",
    "gepa_scoring_matrix":              "threshold",
    "gepa_full":                        "TS gate",
    "gepa_focused_on_difficulty":       "threshold",
    "gepa_gated":                       "TS gate",
}
# Per-mode example selector type (placeholder for ts_batch_size, filled at runtime)
_SEL_TMPL = {
    "gepa_plain_holistic":              "all train",
    "gepa_plain_rubrics":               "all train",
    "gepa_plain_graph":                 "all train",
    "gepa_plain_checklist":             "all train",
    "gepa_plain_instruction_following": "all train",
    "gepa_plain_consistency":           "all train",
    "gepa_plain_comparative":           "all train",
    "gepa_scoring_matrix":              "all train",
    "gepa_full":                        "top {n} (TS)",
    "gepa_focused_on_difficulty":       "top {n} (TS)",
    "gepa_gated":                       "all train",
}


def run_step(baseline_score_holistic: float,
             baseline_score_rubrics: Optional[float],
             *,
             scores: Dict[str, List[float]],
             metrics: Dict[str, Optional[dict]],
             ts_batch_size: int = 4,
             console=None) -> None:
    """Print comparison table for all (mode, fitness_metric) combinations that ran.

    Parameters
    ----------
    scores:
        Dict keyed by run key (e.g. ``"gepa_plain_holistic"`` or ``"gepa_plain_holistic__jiuwen"``).
        Each value is a list of per-run holdout scores.
    metrics:
        Dict keyed by the same run keys; each value is the last-run metrics dict.
    """
    console.print(f"\n[bold cyan]*** Demo Step 07: Compare Results Started ***[/bold cyan]")

    # ── Collect present modes in insertion order ───────────────────────────
    mode_data: list[tuple[str, str, list[float], Optional[dict]]] = []
    # (run_key, display_label, scores_list, metrics_dict)
    for run_key, sc in scores.items():
        if sc:
            mode_data.append((run_key, run_key_label(run_key), sc, metrics.get(run_key)))

    ran_labels = [label for _, label, _, _ in mode_data]
    pre_labels = ["Base-Holistic", "Base-Rubrics"] if baseline_score_rubrics is not None else ["Base-Holistic"]
    all_labels = pre_labels + (ran_labels if ran_labels else ["(pre-training only)"])
    _banner("COMPARISON — " + "  ·  ".join(all_labels), console=console)

    if not mode_data:
        console.print(f"\n  Pre-training single holdout score: {baseline_score_holistic:.4f}  (no training modes ran)")
        if baseline_score_rubrics is not None:
            console.print(f"  Pre-training multi holdout score:  {baseline_score_rubrics:.4f}  (no training modes ran)")
        console.print(f"[bold cyan]*** Demo Step 07: Compare Results Finished - no data ***[/bold cyan]")
        return

    n_runs = len(mode_data[0][2])
    multi = n_runs > 1

    # ── Column widths ──────────────────────────────────────────────────────
    max_label_len = max(len(lbl) for lbl in ran_labels + pre_labels) if ran_labels else 14
    W = max(max_label_len + 2, 13 if multi else 11)

    # Build cols list: (display_label, scores_or_None, metrics_or_None, base_score)
    cols: list[tuple[str, Optional[list[float]], Optional[dict], float]] = [
        ("Base-Holistic", None, None, baseline_score_holistic),
    ]
    if baseline_score_rubrics is not None:
        cols.append(("Base-Rubrics", None, None, baseline_score_rubrics))
    cols += [(label, sc, m, 0.0) for _, label, sc, m in mode_data]

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
    for label, sc, m, base in cols:
        if sc is None:
            score_row += f"  {base:>{W}.4f}"
        elif multi:
            m_val = mean(sc)
            s_val = std(sc)
            score_row += f"  {f'{m_val:.4f} ±{s_val:.4f}':>{W}}"
        else:
            score_row += f"  {sc[0]:>{W}.4f}"
    console.print(score_row)

    # Δ over baseline row
    delta_row = f"  {'Δ over baseline':32s}" + f"  {'—':>{W}}" * len(pre_labels)
    for rk, _, sc2, m2 in mode_data:
        mode_part = run_key_mode(rk)
        mode_baseline = (
            m2.get("baseline_score")
            if m2 and "baseline_score" in m2
            else (
                (baseline_score_rubrics if baseline_score_rubrics is not None else baseline_score_holistic)
                if mode_part == "gepa_plain_rubrics"
                else baseline_score_holistic
            )
        )
        d = mean(sc2) - mode_baseline
        delta_row += f"  {('+' if d >= 0 else '') + f'{d:.4f}':>{W}}"
    console.print(delta_row)

    # Accepted row
    acc_row = f"  {'Accepted (last run)' if multi else 'Accepted':32s}"
    for _, sc, m, _ in cols:
        if sc is None:
            acc_row += f"  {'—':>{W}}"
        else:
            acc_row += f"  {'✓ yes' if m and m.get('accepted') else '✗ no':>{W}}"
    console.print(acc_row)

    # Config rows
    gate_row = f"  {'Acceptance gate':32s}"
    sel_row  = f"  {'Example selector':32s}"
    for (run_key, label, _, _2), (_, sc, _, _3) in zip(
        [(rk, lbl, s, m) for rk, lbl, s, m in mode_data],
        cols[len(pre_labels):],
    ):
        mode_part = run_key_mode(run_key)
        gate_row += f"  {_GATE.get(mode_part, '?'):>{W}}"
        sel_tmpl = _SEL_TMPL.get(mode_part, "?")
        sel_row  += f"  {sel_tmpl.format(n=ts_batch_size):>{W}}"
    gate_prefix = f"  {'Acceptance gate':32s}" + f"  {'—':>{W}}" * len(pre_labels)
    sel_prefix  = f"  {'Example selector':32s}" + f"  {'—':>{W}}" * len(pre_labels)
    console.print(gate_prefix + gate_row[2 + 32:])
    console.print(sel_prefix  + sel_row[2 + 32:])

    # Winner
    accepted_modes = [(label, mean(sc)) for _, label, sc, m in mode_data if m and m.get("accepted")]
    all_modes_scored = [(label, mean(sc)) for _, label, sc, _ in mode_data]

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

    console.print(f"\n  Run-by-run results ({n_runs} independent runs):")
    header2  = f"  {'':22s}"
    divider2 = f"  {'─' * 22}"
    for label, *_ in cols:
        header2  += f"  {label:>{W}}"
        divider2 += f"  {'─' * W}"
    console.print(header2)
    console.print(divider2)

    for i in range(n_runs):
        row = f"  {f'Run {i + 1}':22s}"
        for _, sc, _, base in cols:
            val = base if sc is None else sc[i]
            row += f"  {val:>{W}.4f}"
        console.print(row)

    console.print(f"  {'─' * 22}" + f"  {'─' * W}" * len(cols))
    mean_row = f"  {'Mean':22s}"
    std_row  = f"  {'Std dev':22s}"
    for _, sc, _, base in cols:
        if sc is None:
            mean_row += f"  {base:>{W}.4f}"
            std_row  += f"  {'—':>{W}}"
        else:
            mean_row += f"  {mean(sc):>{W}.4f}"
            std_row  += f"  {std(sc):>{W}.4f}"
    console.print(mean_row)
    console.print(std_row)

    # Bootstrap CI vs first "gepa_plain_holistic" entry (if present and n_runs > 1)
    plain_entries = [(rk, sc) for rk, _, sc, _ in mode_data if run_key_mode(rk) == "gepa_plain_holistic"]
    if plain_entries and len(mode_data) > 1:
        ref_key, ref_scores = plain_entries[0]
        ref_label = run_key_label(ref_key)
        console.print(f"\n  Bootstrap 95% CI vs {ref_label} ({n_runs} runs, paired):")
        for rk, label, sc, _ in mode_data:
            if rk == ref_key:
                continue
            d_mean, lo, hi = bootstrap_ci_diff(ref_scores, sc)
            sign = "+" if d_mean >= 0 else ""
            ci_str = f"[{'+' if lo >= 0 else ''}{lo:.4f}, {'+' if hi >= 0 else ''}{hi:.4f}]"
            if lo > 0:
                verdict = "★ reliably better"
            elif hi < 0:
                verdict = "✗ reliably worse"
            else:
                verdict = "~ inconclusive"
            console.print(f"    {label:<14}  Δ = {sign}{d_mean:.4f}  CI {ci_str}  {verdict}")

    console.print(f"[bold cyan]*** Demo Step 07: Compare Results Finished ***[/bold cyan]")
