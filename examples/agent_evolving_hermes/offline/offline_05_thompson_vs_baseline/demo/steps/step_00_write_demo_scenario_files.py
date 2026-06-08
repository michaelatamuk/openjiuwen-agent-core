
from __future__ import annotations

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_skill import _print_skill
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.writer_golden_dataset import \
    _write_golden_dataset
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.writer_skill import _write_skill


def run_step(skills_root, skill_name, skill_body, skill_frontmatter, examples, verbose: bool = False, console = None):

    # ── Step 1: Write baseline skill + golden dataset ─────────────────────────
    console.print(f"\n[bold cyan]*** Demo Step 00: Write Demo Scenarios Started ***[/bold cyan]")
    skill_path = _write_skill(skills_root, skill_name, skill_frontmatter, skill_body)
    golden_dir = _write_golden_dataset(skills_root, skill_name, examples)
    console.print(f"Pre-training skill: {skill_path}")
    console.print(f"Golden dataset : {golden_dir}")

    if verbose:
        baseline_text = skill_path.read_text()
        _print_skill("① BASELINE SKILL — before any evolution", baseline_text, console)

    console.print(f"[bold cyan]*** Demo Step 00: Write Demo Scenarios Finished ***[/bold cyan]")
