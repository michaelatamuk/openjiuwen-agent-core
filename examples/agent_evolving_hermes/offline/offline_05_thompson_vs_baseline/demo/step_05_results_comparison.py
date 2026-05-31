
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner


def step(metrics_no_ts, metrics_l2, metrics_l3, metrics_l2_l3, TS_BATCH_SIZE):
    _banner("COMPARISON — Baseline  ·  No-TS  ·  L2 only  ·  L3 only  ·  L2+L3")

    bs     = metrics_no_ts.get("baseline_score",  0.0)
    s_no   = metrics_no_ts.get("evolved_score",   0.0)
    s_l2   = metrics_l2.get("evolved_score",      0.0)
    s_l3   = metrics_l3.get("evolved_score",      0.0)
    s_full = metrics_l2_l3.get("evolved_score",   0.0)
    W = 11

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n  {'':32s}  {'Baseline':>{W}}  {'No-TS':>{W}}  "
          f"{'L2 only':>{W}}  {'L3 only':>{W}}  {'L2+L3':>{W}}")
    print(f"  {'─'*32}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}  {'─'*W}")

    # ── Holdout scores ────────────────────────────────────────────────────────
    print(f"  {'Holdout score':32s}  {bs:>{W}.4f}  {s_no:>{W}.4f}  "
          f"{s_l2:>{W}.4f}  {s_l3:>{W}.4f}  {s_full:>{W}.4f}")

    # ── Δ over baseline ───────────────────────────────────────────────────────
    def _delta(s):
        d = s - bs
        return ('+' if d >= 0 else '') + f'{d:.4f}'

    print(f"  {'Δ over baseline':32s}  {'—':>{W}}  "
          f"{_delta(s_no):>{W}}  {_delta(s_l2):>{W}}  "
          f"{_delta(s_l3):>{W}}  {_delta(s_full):>{W}}")

    # ── Accepted ──────────────────────────────────────────────────────────────
    def _acc(m):
        return '✓ yes' if m.get('accepted') else '✗ no'

    print(f"  {'Accepted':32s}  {'—':>{W}}  "
          f"{_acc(metrics_no_ts):>{W}}  {_acc(metrics_l2):>{W}}  "
          f"{_acc(metrics_l3):>{W}}  {_acc(metrics_l2_l3):>{W}}")

    # ── Configuration rows ────────────────────────────────────────────────────
    print(f"  {'Acceptance gate':32s}  {'—':>{W}}  {'threshold':>{W}}  "
          f"{'threshold':>{W}}  {'TS gate':>{W}}  {'TS gate':>{W}}")
    print(f"  {'Example selector':32s}  {'—':>{W}}  {'all train':>{W}}  "
          f"{f'top {TS_BATCH_SIZE} (TS)':>{W}}  {'all train':>{W}}  "
          f"{f'top {TS_BATCH_SIZE} (TS)':>{W}}")
    print(f"  {'Hard examples targeted':32s}  {'—':>{W}}  {'no':>{W}}  "
          f"{'yes':>{W}}  {'no':>{W}}  {'yes':>{W}}")

    # ── Winner ────────────────────────────────────────────────────────────────
    scores = {"No-TS": s_no, "L2 only": s_l2, "L3 only": s_l3, "L2+L3": s_full}
    winner = max(scores, key=lambda k: scores[k])
    best   = scores[winner]
    ties   = [k for k, v in scores.items() if v == best]
    if len(ties) > 1:
        winner_str = " = ".join(ties) + "  (tie)"
    else:
        winner_str = winner

    print(f"\n  ▶  Best evolution: {winner_str}  (holdout score {best:.4f})")
