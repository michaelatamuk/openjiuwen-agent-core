
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.writer_golden_dataset import \
    _write_golden_dataset
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.writer_skill import _write_skill


def step(skills_root, SKILL_NAME, SKILL_BODY, SKILL_FRONTMATTER, GOLDEN_EXAMPLES,
         verbose: bool = False) -> None:

    # ── Step 1: Write baseline skill + golden dataset ─────────────────────────
    skill_path = _write_skill(skills_root, SKILL_NAME, SKILL_FRONTMATTER, SKILL_BODY)
    golden_dir = _write_golden_dataset(skills_root, SKILL_NAME, GOLDEN_EXAMPLES)
    print(f"\nBaseline skill    : {skill_path}")
    print(f"Golden dataset    : {golden_dir}")

    if verbose:
        baseline_text = skill_path.read_text()
        _print_skill("① BASELINE SKILL — before any evolution", baseline_text)
