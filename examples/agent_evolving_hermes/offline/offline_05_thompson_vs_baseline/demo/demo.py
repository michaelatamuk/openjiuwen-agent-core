
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_01_save_skill_and_dataset import \
    step as step_01_save_skill_and_dataset
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_02_run_gepa_without_thompson_sampling import \
    step as step_02_run_gepa_without_thompson_sampling
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_03_restore_baseline_skill import \
    step as step_03_restore_baseline_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_04b_run_gepa_l2_only import \
    step as step_04b_run_gepa_l2_only
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_04c_run_gepa_l3_only import \
    step as step_04c_run_gepa_l3_only
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_04_run_gepa_with_thompson_sampling import \
    step as step_04_run_gepa_with_thompson_sampling
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_05_results_comparison import \
    step as step_05_comparison
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.step_06_final_prints import \
    step as step_06_final_prints


# ══════════════════════════════════════════════════════════════════════════════
# Main demo
# ══════════════════════════════════════════════════════════════════════════════

def run_demo(skills_root, output_no_ts, output_l2_only, output_l3_only, output_l2_l3, ts_state_dir,
             SKILL_NAME, SKILL_BODY, SKILL_FRONTMATTER, ITERATIONS,
             TS_BATCH_SIZE, MODEL, GOLDEN_EXAMPLES, verbose: bool = False) -> None:

    # ── Step 1: Write baseline skill + golden dataset ─────────────────────────
    step_01_save_skill_and_dataset(skills_root, SKILL_NAME, SKILL_BODY, SKILL_FRONTMATTER, GOLDEN_EXAMPLES)

    # ── Step 2: GEPA run — No TS ──────────────────────────────────────────────
    metrics_no_ts = step_02_run_gepa_without_thompson_sampling(
        skills_root, SKILL_NAME, MODEL, ITERATIONS, output_no_ts, verbose=verbose)

    # ── Step 3: Restore baseline skill ───────────────────────────────────────
    step_03_restore_baseline_skill(skills_root, SKILL_NAME, SKILL_FRONTMATTER, SKILL_BODY)

    # ── Step 4b: GEPA run — L2 only (Example Selector, no Acceptance Gate) ───
    metrics_l2 = step_04b_run_gepa_l2_only(
        skills_root, SKILL_NAME, MODEL, ITERATIONS, TS_BATCH_SIZE,
        output_l2_only, ts_state_dir, verbose=verbose)

    # ── Step 3: Restore baseline skill ───────────────────────────────────────
    step_03_restore_baseline_skill(skills_root, SKILL_NAME, SKILL_FRONTMATTER, SKILL_BODY)

    # ── Step 4c: GEPA run — L3 only (Acceptance Gate, all examples) ──────────
    metrics_l3 = step_04c_run_gepa_l3_only(
        skills_root, SKILL_NAME, MODEL, ITERATIONS,
        output_l3_only, ts_state_dir, verbose=verbose)

    # ── Step 3: Restore baseline skill ───────────────────────────────────────
    step_03_restore_baseline_skill(skills_root, SKILL_NAME, SKILL_FRONTMATTER, SKILL_BODY)

    # ── Step 4: GEPA run — L2 + L3 (full Thompson Sampling) ──────────────────
    metrics_l2_l3 = step_04_run_gepa_with_thompson_sampling(
        skills_root, SKILL_NAME, MODEL, ITERATIONS, TS_BATCH_SIZE,
        GOLDEN_EXAMPLES, output_l2_l3, ts_state_dir, verbose=verbose)

    # ── Step 5: Five-way comparison table ────────────────────────────────────
    step_05_comparison(metrics_no_ts, metrics_l2, metrics_l3, metrics_l2_l3, TS_BATCH_SIZE)

    # ── Step 6: Where to look ─────────────────────────────────────────────────
    step_06_final_prints(SKILL_NAME, output_no_ts, output_l2_only, output_l3_only, output_l2_l3, ts_state_dir)

