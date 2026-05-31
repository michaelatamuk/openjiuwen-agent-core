
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.helpers.printer_banner import _banner
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.helpers.printer_metrics import \
    _print_metrics
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.helpers.printer_ts_insights import \
    _print_ts_insights
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.helpers.reader_latest_evolved import \
    _read_latest_evolved
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.helpers.writer_golden_dataset import \
    _write_golden_dataset
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.helpers.writer_skill import _write_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_body import SKILL_BODY
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_frontmatter import SKILL_FRONTMATTER
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, evolve_single_skill


# ══════════════════════════════════════════════════════════════════════════════
# Main demo
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(skills_root, output_no_ts, output_ts, ts_state_dir, SKILL_NAME, ITERATIONS, TS_BATCH_SIZE,
             MODEL, GOLDEN_EXAMPLES) -> None:

    # ── Step 1: Write baseline skill + golden dataset ─────────────────────────
    skill_path = _write_skill(skills_root, SKILL_NAME, SKILL_FRONTMATTER, SKILL_BODY)
    golden_dir = _write_golden_dataset(skills_root, SKILL_NAME, GOLDEN_EXAMPLES)
    print(f"\nBaseline skill    : {skill_path}")
    print(f"Golden dataset    : {golden_dir}")

    baseline_text = skill_path.read_text()
    _print_skill("① BASELINE SKILL — before any evolution", baseline_text)

    # ── Step 2: GEPA run without Thompson Sampling ────────────────────────────
    _banner("② GEPA — without Thompson Sampling")
    print("  Example selector : all training examples, equal weight")
    print("  Acceptance gate  : threshold only (improvement ≥ 0.0)")
    print()

    config_no_ts = EvolverConfig(
        skills_root  = skills_root,
        output_dir   = output_no_ts,
        iterations   = ITERATIONS,
        optimizer_model = MODEL,
        eval_model      = MODEL,
        # Thompson Sampling — all OFF
        ts_skill_scheduler   = False,
        ts_example_selector  = False,
        ts_acceptance_gate   = False,
    )
    metrics_no_ts = evolve_single_skill(
        SKILL_NAME, "golden", config=config_no_ts, min_improvement=0.0
    )
    _print_metrics(metrics_no_ts)

    evolved_no_ts = _read_latest_evolved(output_no_ts, SKILL_NAME)
    _print_skill("  Evolved skill (no TS)", evolved_no_ts or "[not produced]")

    # ── Step 3: Restore baseline skill ───────────────────────────────────────
    _write_skill(skills_root, SKILL_NAME, SKILL_FRONTMATTER, SKILL_BODY)

    # ── Step 4: GEPA run WITH Thompson Sampling (Level 2 + Level 3) ──────────
    _banner("③ GEPA — with Thompson Sampling (Level 2 + Level 3)")
    print(f"  Level 2 — Example Selector  : selects top {TS_BATCH_SIZE} of "
          f"{int(len(GOLDEN_EXAMPLES)*0.5)} train examples per iteration")
    print("    TS learns which examples best distinguish good vs bad evolved skills")
    print("    → hard examples (security, concurrency) accumulate higher α")
    print()
    print(f"  Level 3 — Acceptance Gate   : P(candidate > deployed) ≥ 0.75")
    print("    Monte Carlo (100 draws) — prevents accepting a lucky one-off run")
    print()

    config_ts = EvolverConfig(
        skills_root  = skills_root,
        output_dir   = output_ts,
        iterations   = ITERATIONS,
        optimizer_model = MODEL,
        eval_model      = MODEL,
        # Thompson Sampling — Level 2 + Level 3 ON
        ts_skill_scheduler   = False,          # L1 only matters for --all runs
        ts_example_selector  = True,
        ts_example_batch_size = TS_BATCH_SIZE,
        ts_acceptance_gate   = True,
        ts_acceptance_confidence = 0.75,
        ts_acceptance_n_samples  = 100,
        ts_state_dir = ts_state_dir,
    )
    metrics_ts = evolve_single_skill(
        SKILL_NAME, "golden", config=config_ts, min_improvement=0.0
    )
    _print_metrics(metrics_ts)
    _print_ts_insights(ts_state_dir, SKILL_NAME)

    evolved_ts = _read_latest_evolved(output_ts, SKILL_NAME)
    _print_skill("  Evolved skill (with TS)", evolved_ts or "[not produced]")

    # ── Step 5: Three-way comparison table ───────────────────────────────────
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

    # ── Step 6: Where to look ─────────────────────────────────────────────────
    print(f"\n  Output files:")
    print(f"    No-TS run   →  {output_no_ts}/{SKILL_NAME}/")
    print(f"    With-TS run →  {output_ts}/{SKILL_NAME}/")
    print(f"    TS state    →  {ts_state_dir}/")
    print()
    print("  To inspect TS arm state directly:")
    print(f"    python -m json.tool {ts_state_dir}/ts_examples_{SKILL_NAME}.json")
    print(f"    python -m json.tool {ts_state_dir}/ts_gate_{SKILL_NAME}.json")
    print()
    print("  To re-run with your own skill, edit SKILL_FRONTMATTER / SKILL_BODY")
    print("  and add your own entries to GOLDEN_EXAMPLES at the top of this file.")
