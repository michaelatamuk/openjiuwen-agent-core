
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo import step_01_save_skill_and_dataset
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_01_save_skill_and_dataset import \
    step as step_01_save_skill_and_dataset
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_02_run_gepa_without_thompson_sampling import \
    step as step_02_run_gepa_without_thompson_sampling
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_03_restore_baseline_skill import \
    step as step_03_restore_baseline_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_04_run_gepa_with_thompson_sampling import \
    step as step_04_run_gepa_with_thompson_sampling
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_05_three_way_comparison import \
    step as step_05_three_way_comparison
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_06_final_prints import (
    step as step_06_final_prints)


# ══════════════════════════════════════════════════════════════════════════════
# Main demo
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(skills_root, output_no_ts, output_ts, ts_state_dir, SKILL_NAME, SKILL_BODY, SKILL_FRONTMATTER, ITERATIONS,
             TS_BATCH_SIZE, MODEL, GOLDEN_EXAMPLES, verbose: bool = False) -> None:

    # ── Step 1: Write baseline skill + golden dataset ─────────────────────────
    step_01_save_skill_and_dataset(skills_root, SKILL_NAME, SKILL_BODY, SKILL_FRONTMATTER, GOLDEN_EXAMPLES)

    # ── Step 2: GEPA run without Thompson Sampling ────────────────────────────
    metrics_no_ts = step_02_run_gepa_without_thompson_sampling(skills_root, SKILL_NAME, MODEL, ITERATIONS, output_no_ts,
                                                               verbose=verbose)

    # ── Step 3: Restore baseline skill ───────────────────────────────────────
    step_03_restore_baseline_skill(skills_root, SKILL_NAME, SKILL_FRONTMATTER, SKILL_BODY)

    # ── Step 4: GEPA run WITH Thompson Sampling (Level 2 + Level 3) ──────────
    metrics_ts = step_04_run_gepa_with_thompson_sampling(skills_root, SKILL_NAME, MODEL, ITERATIONS, TS_BATCH_SIZE,
                                                         GOLDEN_EXAMPLES, output_ts, ts_state_dir, verbose=verbose)

    # ── Step 5: Three-way comparison table ───────────────────────────────────
    step_05_three_way_comparison(metrics_no_ts, metrics_ts, TS_BATCH_SIZE)

    # ── Step 6: Where to look ─────────────────────────────────────────────────
    step_06_final_prints(SKILL_NAME, output_no_ts, output_ts, ts_state_dir)

