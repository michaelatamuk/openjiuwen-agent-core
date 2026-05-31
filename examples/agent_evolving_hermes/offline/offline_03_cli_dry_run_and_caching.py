# coding: utf-8
"""Example 7: CLI --dry-run and --reuse-dataset flags.

New CLI flags added to match Hermes evolve_skill CLI:

--dry-run
    Validates the skill exists and passes baseline constraints.
    Prints exactly what WOULD happen without making any LLM calls.
    Zero cost. Use before a real run to catch config errors.

    Mirrors Hermes evolve_skill.py lines 73-78:
        if dry_run:
            console.print("DRY RUN — setup validated successfully.")
            console.print("  Would generate eval dataset ...")
            return

--reuse-dataset
    On re-runs, skips synthetic/external dataset generation and
    reuses the most recently saved dataset from the output directory.
    Saves LLM cost when iterating on optimizer_model or iterations.

python -m invocation (via __main__.py):
    python -m openjiuwen.agent_evolving_hermess --skill git-review --dry-run

Usage:
    python examples/agent_evolving_hermess/07_cli_dry_run_and_caching.py
"""
from click.testing import CliRunner
from openjiuwen.agent_evolving_hermes.offline.cli import main


def show_dry_run_output():
    """Run the CLI in --dry-run mode and show its output."""
    print("── CLI --dry-run mode ───────────────────────────────────────")
    runner = CliRunner()
    result = runner.invoke(main, [
        "--skill", "git-review",
        "--iterations", "5",
        "--optimizer-model", "openai/gpt-4o-mini",
        "--dry-run",
    ])
    # Will show "Skill not found" if git-review doesn't exist —
    # that's expected in this demo. The point is to show the dry-run output.
    print(result.output)
    print(f"  Exit code: {result.exit_code}")
    print()


def show_help():
    """Print the full CLI help including new flags."""
    print("── CLI --help output ────────────────────────────────────────")
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    print(result.output)


def show_reuse_dataset_flag():
    """Demonstrate --reuse-dataset flag behavior (documentation only)."""
    print("── --reuse-dataset flag ─────────────────────────────────────")
    print(
        "When --reuse-dataset is set, evolve() scans for a cached dataset\n"
        "in the output directory before generating a new one:\n\n"
        "  skill_evolver_output/\n"
        "    git-review/\n"
        "      20260527_143021/         ← previous run\n"
        "        dataset/\n"
        "          train.jsonl          ← will be REUSED\n"
        "          val.jsonl\n"
        "          holdout.jsonl\n"
        "        metrics.json\n"
        "        evolved_skill.md\n"
        "      20260528_091500/         ← new run (reuse_dataset=True)\n"
        "        dataset/               ← NOT created (dataset reused)\n"
        "        metrics.json\n"
        "        evolved_skill.md\n\n"
        "CLI usage:\n"
        "  python -m openjiuwen.agent_evolving_hermess \\\n"
        "      --skill git-review \\\n"
        "      --iterations 20 \\\n"
        "      --reuse-dataset\n"
    )


def show_python_m_invocation():
    """Show python -m usage."""
    print("── python -m openjiuwen.agent_evolving_hermess ──────────────")
    print("The package now has a __main__.py so you can invoke it directly:\n")
    print("  python -m openjiuwen.agent_evolving_hermess --skill git-review --dry-run")
    print("  python -m openjiuwen.agent_evolving_hermess --skill git-review --iterations 10")
    print("  python -m openjiuwen.agent_evolving_hermess --help")
    print()


if __name__ == "__main__":
    show_dry_run_output()
    show_help()
    show_reuse_dataset_flag()
    show_python_m_invocation()
