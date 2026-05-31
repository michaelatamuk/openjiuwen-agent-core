from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from openjiuwen.agent_evolving_hermes.offline.config import EvolverConfig


def save_outputs(
    skill_name: str,
    ts: str,
    baseline_score: float,
    evolved_score: float,
    improvement: float,
    accepted: bool,
    min_improvement: float,
    cross_run_delta: Optional[float],
    prior_metrics: Optional[dict],
    config: EvolverConfig,
    optimizer_name: str,
    eval_source: str,
    cached_path: Optional[Path],
    skill_raw: str,
    evolved_text: str,
    evolved_checks: List,
    elapsed: float,
    output_dir: Path,
    console,
    ts_confidence: Optional[float] = None,
) -> dict:
    """Write all output artefacts and return the metrics dict.

    Writes:
      - metrics.json              — full run metrics
      - evolved_skill.md          — only when accepted=True
      - baseline_skill.md         — always
      - metrics_history.jsonl     — appended at skill level (cross-run log)
    """
    metrics = {
        "skill_name": skill_name,
        "timestamp": ts,
        "baseline_score": round(baseline_score, 4),
        "evolved_score": round(evolved_score, 4),
        "improvement": round(improvement, 4),
        "accepted": accepted,
        "min_improvement_threshold": min_improvement,
        "cross_run_delta": cross_run_delta,
        "prior_run_timestamp": prior_metrics.get("timestamp") if prior_metrics else None,
        "iterations": config.iterations,
        "optimizer": optimizer_name,
        "eval_source": eval_source,
        "reused_dataset": cached_path is not None,
        "baseline_chars": len(skill_raw),
        "evolved_chars": len(evolved_text),
        "elapsed_seconds": round(elapsed, 1),
        "ts_acceptance_confidence": round(ts_confidence, 4) if ts_confidence is not None else None,
        "constraint_checks": [
            {"name": c.constraint_name, "passed": c.passed, "message": c.message}
            for c in evolved_checks
        ],
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if accepted:
        (output_dir / "evolved_skill.md").write_text(evolved_text)
    (output_dir / "baseline_skill.md").write_text(skill_raw)

    _append_metrics_history(skill_name, config.output_dir, metrics)

    console.print(f"\n[dim]Outputs saved to {output_dir}[/dim]")
    if accepted:
        console.print(
            f"[dim]History appended to "
            f"{config.output_dir / skill_name / 'metrics_history.jsonl'}[/dim]"
        )
    return metrics


def _append_metrics_history(skill_name: str, output_dir: Path, metrics: dict) -> None:
    """Append metrics to the per-skill metrics_history.jsonl file.

    This file accumulates one JSON record per run so regression trends
    can be plotted or scanned programmatically.  It lives at:
        <output_dir>/<skill_name>/metrics_history.jsonl
    """
    history_path = output_dir / skill_name / "metrics_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")
