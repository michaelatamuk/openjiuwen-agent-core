"""Step: save the scoring matrix dict as a JSON file.

The file is written atomically (write to .tmp, then rename) so a crash
during serialisation does not leave a corrupt file.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.analyze_scoring_matrix import (
    analyze,
)


def run_step(
    matrix_summary: dict,
    skill_name: str,
    output_dir: Path,
    model: str,
    console,
    oracle_data_dir: Optional[str] = None,
) -> Path:
    """Serialise *matrix_summary* to ``scoring_matrix_<timestamp>.json``.

    Parameters
    ----------
    matrix_summary:
        The dict returned by :func:`step_matrix_run_gepa.run_step`.
        Must contain keys ``"matrix"`` and ``"fitness_metrics"``.
    skill_name:
        Used as a metadata field in the JSON output.
    output_dir:
        Directory where the file is written.  Created if absent.
    model:
        LLM model string; stored as metadata.
    console:
        Rich console for progress messages.

    Returns
    -------
    Path
        Absolute path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scoring_matrix_{timestamp}.json"
    out_path = output_dir / filename
    tmp_path = out_path.with_suffix(".json.tmp")

    payload = {
        "run_id": matrix_summary.get("run_id"),
        "skill_name": skill_name,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "fitness_metrics": matrix_summary.get("fitness_metrics", []),
        "skill_metadata": matrix_summary.get("skill_metadata", {}),
        "summary": {
            "evolved_score": matrix_summary.get("evolved_score"),
            "baseline_score": matrix_summary.get("baseline_score"),
            "improvement": matrix_summary.get("improvement"),
            "accepted": matrix_summary.get("accepted"),
        },
        "matrix": matrix_summary.get("matrix", {}),
        "cross_eval": matrix_summary.get("cross_eval", []),
        "baseline_cross_eval": matrix_summary.get("baseline_cross_eval", []),
        "evolved_cross_eval": matrix_summary.get("evolved_cross_eval", []),
    }

    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)
        os.replace(tmp_path, out_path)
    except Exception as exc:  # noqa: BLE001
        tmp_path.unlink(missing_ok=True)
        console.print(f"[red]Failed to save scoring matrix: {exc}[/red]")
        raise

    total_calls   = sum(len(v.get("calls", [])) for v in payload["matrix"].values())
    n_cross       = len(payload.get("cross_eval", []))
    n_baseline_ce = len(payload.get("baseline_cross_eval", []))
    n_evolved_ce  = len(payload.get("evolved_cross_eval", []))
    console.print(f"\n  [green]✓ Scoring matrix saved:[/green] {out_path}")
    console.print(
        f"    {len(payload['matrix'])} metric(s)  ·  {total_calls} calls"
        f"  ·  cross_eval={n_cross}  baseline_ce={n_baseline_ce}  evolved_ce={n_evolved_ce}"
    )

    analyze(out_path)

    # ── Copy to persistent oracle data dir ────────────────────────────────
    if oracle_data_dir:
        oracle_dir = Path(oracle_data_dir).expanduser()
        oracle_dir.mkdir(parents=True, exist_ok=True)
        oracle_dest = oracle_dir / out_path.name
        shutil.copy2(out_path, oracle_dest)
        console.print(f"  [dim]✓ Oracle copy → {oracle_dest}[/dim]")

    return out_path


def _json_default(obj):
    """Fallback serialiser for non-standard types."""
    if hasattr(obj, "__float__"):
        return float(obj)
    if hasattr(obj, "__int__"):
        return int(obj)
    return str(obj)
