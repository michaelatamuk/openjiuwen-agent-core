# coding: utf-8
"""Load and normalise the 3-D skill-scoring matrix from oracle data.

The matrix has axes:
  X – skill_name   (one value per evolved skill in the oracle dir)
  Y – example_id   (training prompt used during GEPA)
  Z – metric_name  (bag_of_words, f1, graph, …)
  V – norm score   (row-normalised so metric scores sum to 1 per example row)

Two file formats are supported transparently:

  Real GEPA output  — ``scoring_matrix_*.json`` files produced by GEPA runs.
                       These have ``"matrix"`` and ``"cross_eval"`` top-level keys.
                       ``cross_eval`` holds per-example, per-candidate scores across
                       all metrics (x=skill, y=prompt, z=metric → true 3-D matrix).
                       All ``cross_eval`` rows are loaded; the *variant* parameter
                       is ignored for this format.

  Synthetic benchmark oracle — files produced by the benchmark data loaders in
                       this package (``data_loaders/``).  These have
                       ``"baseline_cross_eval"`` / ``"evolved_cross_eval"`` keys.
                       The *variant* parameter selects which key to read.

Loaded data is returned as a flat pandas DataFrame:
  run_id | skill_name | example_id | example_input | score_<m> | norm_<m> | …
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _find_files(oracle_dir: Path) -> list[Path]:
    return sorted(oracle_dir.glob("scoring_matrix_*.json"))


def _is_gepa_format(data: dict) -> bool:
    """Return True when *data* is a real GEPA output file.

    GEPA files always have both a ``"matrix"`` dict (per-metric summary with
    individual ``"calls"``) and a ``"cross_eval"`` list (per-example, per-
    candidate scores across every metric).  Synthetic benchmark oracle files
    produced by this package use ``"baseline_cross_eval"`` instead.
    """
    return bool(data.get("matrix")) and "cross_eval" in data


def _load_gepa(data: dict, path: Path) -> list[dict]:
    """Load rows from the ``cross_eval`` section of a real GEPA output file.

    Each ``cross_eval`` entry is one (candidate, example) pair scored against
    all metrics — the direct 3-D matrix the recommender needs.  All rows are
    kept so the embedder has the most signal; the recommender aggregates them
    with similarity-weighted averaging.
    """
    run_id      = data.get("run_id", path.stem)
    skill_name  = data.get("skill_name", "")
    all_metrics = data.get("fitness_metrics") or []

    rows = []
    for row in data.get("cross_eval") or []:
        scores = row.get("scores") or {}
        entry: dict = {
            "run_id":           run_id,
            "skill_name":       skill_name,
            "example_id":       row.get("example_id", ""),
            "example_input":    row.get("example_input", ""),
            "example_expected": (row.get("example_expected") or "")[:300],
            "candidate_output": (row.get("candidate_output") or "")[:300],
            # Keep provenance fields present in GEPA cross_eval
            "source_metric":    row.get("source_metric", ""),
            "candidate_idx":    row.get("candidate_idx"),
            "gepa_iteration":   row.get("gepa_iteration"),
        }
        for m in all_metrics:
            entry[f"score_{m}"] = scores.get(m)
        rows.append(entry)
    return rows


def _load_benchmark(data: dict, path: Path, key: str) -> list[dict]:
    """Load rows from ``baseline_cross_eval`` or ``evolved_cross_eval``."""
    run_id      = data.get("run_id", path.stem)
    skill_name  = data.get("skill_name", "")
    all_metrics = data.get("fitness_metrics") or []

    rows = []
    for row in data.get(key) or []:
        scores = row.get("scores") or {}
        entry: dict = {
            "run_id":           run_id,
            "skill_name":       skill_name,
            "example_id":       row.get("example_id", ""),
            "example_input":    row.get("example_input", ""),
            "example_expected": (row.get("example_expected") or "")[:300],
            "candidate_output": (row.get("candidate_output") or "")[:300],
        }
        if key == "evolved_cross_eval":
            entry["source_metric"] = row.get("source_metric", "")
        for m in all_metrics:
            entry[f"score_{m}"] = scores.get(m)
        rows.append(entry)
    return rows


def _load_one(path: Path, key: str) -> list[dict]:
    """Return flattened rows from one scoring_matrix_*.json.

    Auto-detects format: real GEPA files are read from ``cross_eval``; synthetic
    benchmark files are read from the key specified by the caller.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print(f"  [WARN] Cannot read {path.name}: {exc}", file=sys.stderr)
        return []

    if _is_gepa_format(data):
        return _load_gepa(data, path)
    return _load_benchmark(data, path, key)


def _normalise_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``norm_<metric>`` columns so metric scores sum to 1 per row."""
    score_cols = [c for c in df.columns if c.startswith("score_")]
    if not score_cols:
        return df
    row_sums = df[score_cols].clip(lower=0).sum(axis=1).replace(0, float("nan"))
    for col in score_cols:
        metric = col[len("score_"):]
        df[f"norm_{metric}"] = df[col].clip(lower=0) / row_sums
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_scores_matrix(
    oracle_dir: Path,
    variant: str = "baseline",
) -> pd.DataFrame:
    """Load the 3-D skill matrix from all scoring_matrix_*.json in *oracle_dir*.

    Parameters
    ----------
    oracle_dir : Path
        Directory containing ``scoring_matrix_*.json`` files.
    variant : str
        Applies only to **synthetic benchmark** files (those produced by
        ``data_loaders/``).  Ignored for real GEPA output files.

        ``"baseline"`` — unchanged seed skill outputs (default)
        ``"evolved"``  — best-candidate (proxy evolved skill) outputs
        ``"both"``     — union of baseline and evolved rows

    Returns
    -------
    pd.DataFrame  with columns:
        run_id, skill_name, example_id, example_input, example_expected,
        candidate_output, [source_metric,] [candidate_idx,] [gepa_iteration,]
        score_<m>, norm_<m>, …

        GEPA files add ``source_metric``, ``candidate_idx``, ``gepa_iteration``
        columns.  Benchmark files add ``source_metric`` only for evolved rows.
    """
    oracle_dir = Path(oracle_dir).expanduser()
    if not oracle_dir.exists():
        raise FileNotFoundError(f"Oracle dir not found: {oracle_dir}")

    files = _find_files(oracle_dir)
    if not files:
        raise FileNotFoundError(f"No scoring_matrix_*.json found in {oracle_dir}")

    key_map = {
        "baseline": ["baseline_cross_eval"],
        "evolved":  ["evolved_cross_eval"],
        "both":     ["baseline_cross_eval", "evolved_cross_eval"],
    }
    keys = key_map.get(variant)
    if keys is None:
        raise ValueError(f"variant must be 'baseline', 'evolved', or 'both'; got {variant!r}")

    all_rows: list[dict] = []
    for path in files:
        for key in keys:
            all_rows.extend(_load_one(path, key))

    if not all_rows:
        raise ValueError(
            f"No cross-eval data found in {oracle_dir}.\n"
            "Expected either:\n"
            "  • Real GEPA output files with a 'cross_eval' key, OR\n"
            "  • Benchmark oracle files with a 'baseline_cross_eval' key.\n"
            "Run gepa_scoring_matrix to generate GEPA output, or use\n"
            "'python runner.py --benchmarks' to generate benchmark oracle files."
        )

    df = pd.DataFrame(all_rows)
    df = _normalise_rows(df)
    return df.reset_index(drop=True)


def metric_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of metric names found in *df* (from score_* columns)."""
    return [c[len("score_"):] for c in df.columns if c.startswith("score_")]


def norm_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of norm_* column names in *df*."""
    return [c for c in df.columns if c.startswith("norm_")]
