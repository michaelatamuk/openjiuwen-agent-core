from __future__ import annotations


def step(SKILL_NAME, output_no_ts, output_l2_only, output_l3_only, output_l2_l3, ts_state_dir):
    print(f"\n  Output files:")
    print(f"    No-TS run    →  {output_no_ts}/{SKILL_NAME}/")
    print(f"    L2-only run  →  {output_l2_only}/{SKILL_NAME}/")
    print(f"    L3-only run  →  {output_l3_only}/{SKILL_NAME}/")
    print(f"    L2+L3 run    →  {output_l2_l3}/{SKILL_NAME}/")
    print(f"    TS state     →  {ts_state_dir}/")
    print()
    print("  To inspect TS arm state directly:")
    print(f"    python -m json.tool {ts_state_dir}/ts_examples_{SKILL_NAME}.json")
    print(f"    python -m json.tool {ts_state_dir}/ts_gate_{SKILL_NAME}.json")
    print()
    print("  To re-run with your own skill, edit SKILL_FRONTMATTER / SKILL_BODY")
    print("  and add your own entries to GOLDEN_EXAMPLES at the top of this file.")
