from __future__ import annotations

import difflib
import shutil
from typing import Optional

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner

_NUM_W  = 4   # " 12 " — line-number field width (handles up to line 999)
_MARK_W = 2   # "< " / "> " / "  " — diff-marker field width


def print_skill_diff(baseline_text: str,
                     winner_label: str,
                     winner_text: str,
                     winner_score: Optional[float] = None,
                     console=None
) -> None:
    """Print baseline and winner skills side by side with diff alignment.

    Each column shows line numbers.  Changed lines are prefixed ``< `` /
    ``> ``.  Blank lines that are part of a change block are rendered as
    ``·`` so they are not invisible.  A full-width separator with a plain-
    English description is drawn before every change block.  A summary
    (line count, char count, Δ, block count) is printed above the diff.
    """
    score_suffix = f"  (score: {winner_score:.4f})" if winner_score is not None else ""
    _banner(f"SKILL DIFF — Baseline  ·  {winner_label}{score_suffix}", console=console)

    term_w = max(80, shutil.get_terminal_size((120, 40)).columns)
    sep    = " │ "
    col_w  = (term_w - len(sep)) // 2
    cont_w = col_w - _NUM_W - _MARK_W   # usable content chars per column

    baseline_lines = baseline_text.strip().splitlines()
    winner_lines   = winner_text.strip().splitlines()

    b_nlines = len(baseline_lines)
    w_nlines = len(winner_lines)
    b_chars  = len(baseline_text.strip())
    w_chars  = len(winner_text.strip())

    sm      = difflib.SequenceMatcher(None, baseline_lines, winner_lines, autojunk=False)
    opcodes = sm.get_opcodes()
    n_blocks = sum(1 for tag, *_ in opcodes if tag != "equal")

    # ── Summary ───────────────────────────────────────────────────────────────
    dl_sign = "+" if w_nlines >= b_nlines else ""
    dc_sign = "+" if w_chars  >= b_chars  else ""
    dl = f"{dl_sign}{w_nlines - b_nlines}"
    dc = f"{dc_sign}{w_chars  - b_chars}"

    if n_blocks == 0:
        console.print(f"  Baseline  {b_nlines} lines · {b_chars} chars")
        console.print(f"  Winner    {w_nlines} lines · {w_chars} chars")
        console.print()
        console.print("  (skills are identical — no changes were made by GEPA)")
        console.print()
        return

    block_word = "block" if n_blocks == 1 else "blocks"
    console.print(f"  Baseline  {b_nlines} lines · {b_chars} chars")
    console.print(f"  Winner    {w_nlines} lines · {w_chars} chars   "
                  f"Δ {dl} lines · {dc} chars · {n_blocks} change {block_word}")
    console.print()

    # ── Column headers ────────────────────────────────────────────────────────
    def _col_hdr(label: str) -> str:
        inner  = f" {label} "
        dashes = max(0, col_w - len(inner))
        return "─" * (dashes // 2) + inner + "─" * (dashes - dashes // 2)

    console.print(_col_hdr(f"Baseline  [{b_nlines} ln]") + sep + _col_hdr(f"{winner_label}  [{w_nlines} ln]"))
    console.print()

    # ── Helpers ───────────────────────────────────────────────────────────────
    full_w = 2 * col_w + len(sep)

    def _diff_sep(desc: str) -> None:
        """Full-width separator line with an embedded plain-English description."""
        inner  = f" {desc} "
        dashes = max(0, full_w - len(inner))
        console.print("╌" * (dashes // 2) + inner + "╌" * (dashes - dashes // 2))

    def _fit(s: str) -> str:
        if len(s) > cont_w:
            return s[:cont_w - 1] + "…"
        return s.ljust(cont_w)

    def _lnum(n: int) -> str:
        """Right-aligned line number, padded to _NUM_W chars."""
        return f"{n:>{_NUM_W - 1}} " if n > 0 else " " * _NUM_W

    def _visible(s: str, is_active: bool) -> str:
        """Replace blank lines on the active diff side with a visible marker."""
        return "·" if (is_active and s.strip() == "") else s

    def _row(left: str, right: str,
             ml: str = "  ", mr: str = "  ",
             ll: int = 0,    rl: int = 0) -> None:
        is_diff_l = ml != "  "
        is_diff_r = mr != "  "
        lv = _visible(left,  is_diff_l)
        rv = _visible(right, is_diff_r)
        console.print(_lnum(ll) + ml + _fit(lv) + sep + _lnum(rl) + mr + _fit(rv))

    # ── Diff body ─────────────────────────────────────────────────────────────
    ln_left  = 1
    ln_right = 1

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                _row(baseline_lines[i1 + k], winner_lines[j1 + k],
                     ll=ln_left + k, rl=ln_right + k)
            ln_left  += i2 - i1
            ln_right += j2 - j1

        else:
            left_block  = baseline_lines[i1:i2]
            right_block = winner_lines[j1:j2]
            n = max(len(left_block), len(right_block))

            if tag == "delete":
                s = "s" if len(left_block) > 1 else ""
                _diff_sep(f"─ {len(left_block)} line{s} removed")
            elif tag == "insert":
                s = "s" if len(right_block) > 1 else ""
                _diff_sep(f"+ {len(right_block)} line{s} inserted")
            else:
                s = "s" if n > 1 else ""
                _diff_sep(f"~ {n} line{s} changed")

            for k in range(n):
                left  = left_block[k]  if k < len(left_block)  else ""
                right = right_block[k] if k < len(right_block) else ""
                ml    = "< " if k < len(left_block)  else "  "
                mr    = "> " if k < len(right_block) else "  "
                ll_n  = ln_left  + k if k < len(left_block)  else 0
                rl_n  = ln_right + k if k < len(right_block) else 0
                _row(left, right, ml, mr, ll_n, rl_n)

            ln_left  += len(left_block)
            ln_right += len(right_block)

    console.print()
