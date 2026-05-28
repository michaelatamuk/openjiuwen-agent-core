# coding: utf-8
"""Example 10 — Cross-run metrics history and batch evolution.

Demonstrates the new features added to evolve.py and cli.py:

  1. metrics_history.jsonl — per-skill append-log across runs; enables
     regression detection over multiple evolution cycles.
  2. min_improvement acceptance gate — evolved skill not deployed if
     improvement < threshold (saved as evolved_REGRESSION.md instead).
  3. cross_run_delta — improvement vs the prior run's evolved_score.
  4. batch_evolve() — evolve a list of skills sequentially with a single call.
  5. --all CLI flag — evolve all skills under skills-root in one invocation.
  6. --min-improvement CLI flag — pass acceptance threshold from the CLI.

This example uses click.testing.CliRunner and manually-written metrics stubs
so that no LLM calls are made.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from openjiuwen.agent_evolving_hermess.offline.cli import main as cli_main


# ── Helpers ──────────────────────────────────────────────────────────────────

_SKILL_TEMPLATE = """\
---
name: {name}
description: {desc}
---
## Instructions
{body}
"""


def _write_skill(root: Path, name: str, desc: str = "", body: str = "") -> Path:
    """Write a minimal SKILL.md under root/<name>/."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        _SKILL_TEMPLATE.format(
            name=name,
            desc=desc or f"{name} skill",
            body=body or f"Do {name} work.",
        )
    )
    return skill_dir


def _inject_prior_metrics(output_dir: Path, skill_name: str, run_ts: str, data: dict) -> None:
    """Inject a synthetic prior-run metrics.json to simulate cross-run context."""
    run_dir = output_dir / skill_name / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(data, indent=2))
    # Also add to history
    history_path = output_dir / skill_name / "metrics_history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(data) + "\n")


# ── Demo 1: metrics_history.jsonl structure ───────────────────────────────────

def demo_metrics_history() -> None:
    """Show the metrics_history.jsonl format and how to read it."""
    print("\n── Demo 1: metrics_history.jsonl ──────────────────────────────")
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)

        # Simulate 3 evolution runs for "code-review" skill
        runs = [
            {
                "skill_name": "code-review",
                "timestamp": "20260101_100000",
                "baseline_score": 0.60,
                "evolved_score": 0.65,
                "improvement": 0.05,
                "accepted": True,
                "min_improvement_threshold": 0.03,
                "cross_run_delta": None,
            },
            {
                "skill_name": "code-review",
                "timestamp": "20260115_120000",
                "baseline_score": 0.63,
                "evolved_score": 0.70,
                "improvement": 0.07,
                "accepted": True,
                "min_improvement_threshold": 0.03,
                "cross_run_delta": 0.05,  # vs prior evolved_score 0.65
            },
            {
                "skill_name": "code-review",
                "timestamp": "20260201_090000",
                "baseline_score": 0.68,
                "evolved_score": 0.66,
                "improvement": -0.02,
                "accepted": False,  # below threshold 0.03
                "min_improvement_threshold": 0.03,
                "cross_run_delta": -0.04,  # regression vs prior 0.70
            },
        ]

        history_path = output_dir / "code-review" / "metrics_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        for run in runs:
            with open(history_path, "a") as f:
                f.write(json.dumps(run) + "\n")

        # Read and display history
        history = []
        with open(history_path) as f:
            for line in f:
                history.append(json.loads(line.strip()))

        print(f"  History file: {history_path}")
        print(f"  Runs recorded: {len(history)}")
        for h in history:
            accepted_str = "✓" if h["accepted"] else "✗"
            xr = f" | cross_run_delta={h['cross_run_delta']:+.4f}" if h["cross_run_delta"] is not None else ""
            print(
                f"  {accepted_str} [{h['timestamp']}] "
                f"baseline={h['baseline_score']:.4f} "
                f"evolved={h['evolved_score']:.4f} "
                f"Δ={h['improvement']:+.4f}"
                f"{xr}"
            )

        regressions = [h for h in history if not h["accepted"]]
        print(f"\n  Rejected runs (below threshold): {len(regressions)}")
        print(f"  ✓ metrics_history.jsonl demo complete")


# ── Demo 2: --dry-run with --min-improvement ─────────────────────────────────

def demo_cli_min_improvement() -> None:
    """Show --min-improvement flag in dry-run mode (no LLM calls)."""
    print("\n── Demo 2: --min-improvement in dry-run ──────────────────────")
    with tempfile.TemporaryDirectory() as tmp:
        skills_root = Path(tmp) / "skills"
        output_dir = Path(tmp) / "output"
        _write_skill(skills_root, "summarizer", desc="Summarises text documents")

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "--skill", "summarizer",
                "--skills-root", str(skills_root),
                "--output-dir", str(output_dir),
                "--dry-run",
                "--min-improvement", "0.10",
                "--iterations", "5",
            ],
        )
        print(result.output)
        assert result.exit_code == 0, f"Exit code: {result.exit_code}\n{result.output}"
        assert "0.1000" in result.output or "0.10" in result.output
        print("  ✓ --min-improvement shown in dry-run output")


# ── Demo 3: --all flag in dry-run mode ────────────────────────────────────────

def demo_cli_all_flag() -> None:
    """Show --all flag (evolves all skills, dry-run to skip LLM)."""
    print("\n── Demo 3: --all flag in dry-run ─────────────────────────────")
    with tempfile.TemporaryDirectory() as tmp:
        skills_root = Path(tmp) / "skills"
        output_dir = Path(tmp) / "output"
        for name in ("summarizer", "code-reviewer", "translator"):
            _write_skill(skills_root, name)

        runner = CliRunner()
        result = runner.invoke(
            cli_main,
            [
                "--all",
                "--skills-root", str(skills_root),
                "--output-dir", str(output_dir),
                "--dry-run",
                "--min-improvement", "0.05",
            ],
        )
        print(result.output)
        assert result.exit_code == 0, f"Exit code: {result.exit_code}\n{result.output}"
        # All 3 skills should be validated
        assert "summarizer" in result.output
        assert "code-reviewer" in result.output
        assert "translator" in result.output
        print("  ✓ --all flag validated all 3 skills")


# ── Demo 4: mutual exclusion guard ────────────────────────────────────────────

def demo_cli_mutual_exclusion() -> None:
    """Show that --skill and --all together are rejected."""
    print("\n── Demo 4: --skill + --all raises UsageError ─────────────────")
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["--skill", "foo", "--all"],
    )
    print(f"  Exit code: {result.exit_code}")
    print(f"  Output: {result.output.strip()}")
    assert result.exit_code != 0
    print("  ✓ Mutual exclusion correctly enforced")


# ── Demo 5: batch_evolve() API shape ──────────────────────────────────────────

def demo_batch_evolve_api() -> None:
    """Describe the batch_evolve() API without calling it (needs LLM)."""
    print("\n── Demo 5: batch_evolve() API ────────────────────────────────")
    from evolvers.evolve import evolve_skills_batch
    import inspect

    sig = inspect.signature(evolve_skills_batch)
    print(f"  Signature: batch_evolve{sig}")
    print("  Parameters:")
    for name, param in sig.parameters.items():
        default = "" if param.default is inspect.Parameter.empty else f" = {param.default!r}"
        print(f"    {name}: {param.annotation}{default}")
    print(
        "\n  batch_evolve() calls evolve() for each skill_name in sequence.\n"
        "  If a skill raises an exception, the error is captured in the\n"
        "  result dict as {'skill_name': name, 'error': '<message>'}.\n"
        "  All remaining skills still run."
    )
    print("  ✓ batch_evolve API inspection complete")


# ── Run all demos ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_metrics_history()
    demo_cli_min_improvement()
    demo_cli_all_flag()
    demo_cli_mutual_exclusion()
    demo_batch_evolve_api()
    print("\n✓ All cross-run metrics and batch evolve demos passed.")
