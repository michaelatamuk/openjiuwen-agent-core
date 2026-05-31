
from __future__ import annotations

from typing import Optional

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner


def step(
    baseline_score: float,
    *,
    metrics_no_ts:  Optional[dict] = None,
    metrics_l2:     Optional[dict] = None,
    metrics_l3:     Optional[dict] = None,
    metrics_l2_l3:  Optional[dict] = None,
    ts_batch_size:  int = 4,
) -> None:
    """Print a side-by-side comparison for whichever training modes ran.

    ``baseline_score`` is the pre-training holdout score (from step_00).
    Only columns for modes whose metrics dict is non-None are rendered.
    """
    ran = [
        label
        for label, m in [
            ("No-TS",   metrics_no_ts),
            ("L2 only", metrics_l2),
            ("L3 only", metrics_l3),
            ("L2+L3",   metrics_l2_l3),
        ]
        if m is not None
    ]
    _banner("COMPARISON — Baseline  ·  " + ("  ·  ".join(ran) if ran else "(baseline only)"))

    # ── Ordered column list: (header, metrics | None) ─────────────────────
    cols: list[tuple[str, Optional[dict]]] = [("Baseline", None)]
    if metrics_no_ts  is not None: cols.append(("No-TS",   metrics_no_ts))
    if metrics_l2     is not None: cols.append(("L2 only", metrics_l2))
    if metrics_l3     is not None: cols.append(("L3 only", metrics_l3))
    if metrics_l2_l3  is not None: cols.append(("L2+L3",   metrics_l2_l3))

    if len(cols) == 1:
        print(f"\n  Baseline holdout score: {baseline_score:.4f}  (no training modes ran)")
        return

    W = 11

    # ── Header ────────────────────────────────────────────────────────────
    header  = f"\n  {'':32s}"
    divider = f"  {'─'*32}"
    for label, _ in cols:
        header  += f"  {label:>{W}}"
        divider += f"  {'─'*W}"
    print(header)
    print(divider)

    # ── Holdout scores ────────────────────────────────────────────────────
    score_row = f"  {'Holdout score':32s}"
    for _, m in cols:
        val = baseline_score if m is None else m.get("evolved_score", 0.0)
        score_row += f"  {val:>{W}.4f}"
    print(score_row)

    # ── Δ over baseline ───────────────────────────────────────────────────
    delta_row = f"  {'Δ over baseline':32s}"
    for _, m in cols:
        if m is None:
            delta_row += f"  {'—':>{W}}"
        else:
            d = m.get("evolved_score", 0.0) - baseline_score
            delta_row += f"  {('+' if d >= 0 else '') + f'{d:.4f}':>{W}}"
    print(delta_row)

    # ── Accepted ──────────────────────────────────────────────────────────
    acc_row = f"  {'Accepted':32s}"
    for _, m in cols:
        if m is None:
            acc_row += f"  {'—':>{W}}"
        else:
            acc_row += f"  {'✓ yes' if m.get('accepted') else '✗ no':>{W}}"
    print(acc_row)

    # ── Config rows ───────────────────────────────────────────────────────
    _GATE = {
        "No-TS":   "threshold",
        "L2 only": "threshold",
        "L3 only": "TS gate",
        "L2+L3":   "TS gate",
    }
    _SEL = {
        "No-TS":   "all train",
        "L2 only": f"top {ts_batch_size} (TS)",
        "L3 only": "all train",
        "L2+L3":   f"top {ts_batch_size} (TS)",
    }

    gate_row = f"  {'Acceptance gate':32s}"
    sel_row  = f"  {'Example selector':32s}"
    for label, m in cols:
        if m is None:
            gate_row += f"  {'—':>{W}}"
            sel_row  += f"  {'—':>{W}}"
        else:
            gate_row += f"  {_GATE.get(label, '?'):>{W}}"
            sel_row  += f"  {_SEL.get(label, '?'):>{W}}"
    print(gate_row)
    print(sel_row)

    # ── Winner ────────────────────────────────────────────────────────────
    training = {label: m.get("evolved_score", 0.0) for label, m in cols[1:]}
    best_score = max(training.values())
    ties = [k for k, v in training.items() if v == best_score]
    winner_str = (" = ".join(ties) + "  (tie)") if len(ties) > 1 else ties[0]
    print(f"\n  ▶  Best evolution: {winner_str}  (holdout {best_score:.4f})")
