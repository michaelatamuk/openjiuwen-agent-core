from __future__ import annotations

from pathlib import Path


def step(skill_name: str, runs: list[tuple[str, Path]], ts_state_dir: Path) -> None:
    """Print output-file locations for whichever training passes actually ran.

    Parameters
    ----------
    skill_name:
        Skill identifier (sub-directory name used inside each output dir).
    runs:
        List of ``(label, output_dir)`` pairs for the passes that ran.
        Pass ``[]`` when only the baseline evaluation was performed.
    ts_state_dir:
        Directory holding the shared TS arm-state JSON files.
    """
    label_w = max((max(len(l) for l, _ in runs) if runs else 0), 8) + 2

    print("\n  Output files:")
    if runs:
        for label, path in runs:
            print(f"    {label:{label_w}}→  {path}/{skill_name}/")
    else:
        print("    (no training passes ran — baseline evaluation only)")
    print(f"    {'TS state':{label_w}}→  {ts_state_dir}/")

    print()
    print("  To inspect TS arm state directly:")
    print(f"    python -m json.tool {ts_state_dir}/ts_examples_{skill_name}.json")
    print(f"    python -m json.tool {ts_state_dir}/ts_gate_{skill_name}.json")
    print()
    print("  To re-run with your own skill, edit SKILL_FRONTMATTER / SKILL_BODY")
    print("  and add your own entries to GOLDEN_EXAMPLES in the scenario file.")
