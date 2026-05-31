
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner


def step(metrics_no_ts, metrics_ts, TS_BATCH_SIZE):
    _banner("COMPARISON — Baseline  ·  GEPA no-TS  ·  GEPA with-TS")

    bs   = metrics_no_ts.get("baseline_score", 0.0)
    s_no = metrics_no_ts.get("evolved_score",  0.0)
    s_ts = metrics_ts.get("evolved_score",     0.0)
    W = 16

    print(f"\n  {'':32s}  {'Baseline':>{W}}  {'No-TS':>{W}}  {'With-TS':>{W}}")
    print(f"  {'─'*32}  {'─'*W}  {'─'*W}  {'─'*W}")
    print(f"  {'Holdout score':32s}  {bs:>{W}.4f}  {s_no:>{W}.4f}  {s_ts:>{W}.4f}")
    d_no = s_no - bs
    d_ts = s_ts - bs
    print(f"  {'Δ over baseline':32s}  {'—':>{W}}  "
          f"{('+' if d_no>=0 else '') + f'{d_no:.4f}':>{W}}  "
          f"{('+' if d_ts>=0 else '') + f'{d_ts:.4f}':>{W}}")
    accepted_no = '✓ yes' if metrics_no_ts.get('accepted') else '✗ no'
    accepted_ts = '✓ yes' if metrics_ts.get('accepted')    else '✗ no'
    print(f"  {'Accepted':32s}  {'—':>{W}}  {accepted_no:>{W}}  {accepted_ts:>{W}}")
    print(f"  {'Acceptance gate':32s}  {'—':>{W}}  {'threshold':>{W}}  {'TS confidence':>{W}}")
    print(f"  {'Examples / iteration':32s}  {'—':>{W}}  {'all train':>{W}}  "
          f"{f'top {TS_BATCH_SIZE} (TS-ranked)':>{W}}")
    print(f"  {'Hard examples targeted':32s}  {'—':>{W}}  {'no':>{W}}  {'yes (learned)':>{W}}")

    winner = ("With-TS" if s_ts > s_no
              else "No-TS"  if s_no > s_ts
              else "Tie")
    print(f"\n  ▶  Best evolution: {winner}  "
          f"(Δ = {s_ts - s_no:+.4f} in favour of TS run)")
