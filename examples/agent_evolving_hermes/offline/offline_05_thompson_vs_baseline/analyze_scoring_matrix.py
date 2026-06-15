"""Analyze a scoring_matrix_*.json file produced by gepa_scoring_matrix mode.

Usage
-----
# Analyze the latest matrix in a run directory:
    python analyze_scoring_matrix.py <path/to/scoring_matrix_*.json>

# Or run without arguments to pick up the most recent file under /tmp:
    python analyze_scoring_matrix.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.4f}".format)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate(text: str, n: int = 80) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def _find_latest_matrix() -> Path:
    import glob
    files = sorted(glob.glob("/tmp/**/scoring_matrix_*.json", recursive=True) +
                   glob.glob("/var/folders/**/scoring_matrix_*.json", recursive=True))
    if not files:
        sys.exit("No scoring_matrix_*.json found under /tmp or /var/folders. Pass the path explicitly.")
    return Path(files[-1])


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Build flat DataFrame ──────────────────────────────────────────────────────

def build_df(data: dict) -> pd.DataFrame:
    rows = []
    for metric_name, metric_data in data["matrix"].items():
        for call in metric_data.get("calls", []):
            rows.append({
                "metric":          metric_name,
                "candidate_idx":   call.get("candidate_idx"),
                "example_idx":     call.get("example_idx"),
                "call_idx":        call.get("call_idx"),
                "score":           call.get("score"),
                "example_input":   _truncate(call.get("example_input", ""), 70),
                "candidate_output": _truncate(call.get("candidate_output", ""), 70),
            })
    return pd.DataFrame(rows)


# ── Print sections ────────────────────────────────────────────────────────────

def print_header(data: dict, path: Path) -> None:
    s = data["summary"]
    print("=" * 90)
    print(f"  Scoring Matrix Analysis")
    print(f"  Skill   : {data['skill_name']}")
    print(f"  Model   : {data['model']}")
    print(f"  File    : {path.name}")
    print(f"  Metrics : {', '.join(data['fitness_metrics'])}")
    print(f"  Summary : baseline={s['baseline_score']:.4f}  evolved={s['evolved_score']:.4f}"
          f"  Δ={s['improvement']:+.4f}  accepted={s['accepted']}")
    print("=" * 90)


def print_per_metric_meta(data: dict) -> None:
    print("\n── Per-metric overview ──────────────────────────────────────────────────")
    rows = []
    for mn, mv in data["matrix"].items():
        rows.append({
            "metric":                  mn,
            "n_candidates":            mv.get("n_candidates"),
            "examples_per_candidate":  mv.get("n_examples_per_candidate"),
            "total_calls":             len(mv.get("calls", [])),
            "baseline_score":          mv.get("baseline_score"),
            "evolved_score":           mv.get("evolved_score"),
            "improvement":             mv.get("improvement"),
            "accepted":                mv.get("accepted"),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def print_score_distribution(df: pd.DataFrame) -> None:
    print("\n── Score distribution per metric ────────────────────────────────────────")
    stats = (df.groupby("metric")["score"]
               .agg(["count", "mean", "std", "min",
                     lambda x: x.quantile(0.25),
                     "median",
                     lambda x: x.quantile(0.75),
                     "max"])
               .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"}))
    print(stats.to_string())


def print_score_by_candidate(df: pd.DataFrame) -> None:
    print("\n── Mean score per (metric, candidate_idx) ───────────────────────────────")
    pivot = (df.groupby(["metric", "candidate_idx"])["score"]
               .agg(["mean", "std", "count"])
               .rename(columns={"mean": "score_mean", "std": "score_std", "count": "n_evals"}))
    print(pivot.to_string())


def print_score_by_example(df: pd.DataFrame) -> None:
    print("\n── Mean score per (metric, example_idx) — which examples are harder? ───")
    pivot = (df.groupby(["metric", "example_idx"])["score"]
               .agg(["mean", "std", "count"])
               .rename(columns={"mean": "score_mean", "std": "score_std", "count": "n_evals"}))
    print(pivot.to_string())


def print_cross_metric_per_example(df: pd.DataFrame) -> None:
    print("\n── Cross-metric score per example_idx (mean across all candidates) ─────")
    pivot = (df.groupby(["example_idx", "metric"])["score"]
               .mean()
               .unstack("metric"))
    pivot["row_mean"] = pivot.mean(axis=1)
    print(pivot.to_string())


def print_candidate_progress(df: pd.DataFrame) -> None:
    print("\n── Score progression by candidate_idx (did GEPA improve over time?) ────")
    pivot = (df.groupby(["candidate_idx", "metric"])["score"]
               .mean()
               .unstack("metric"))
    pivot["row_mean"] = pivot.mean(axis=1)
    print(pivot.to_string())


def print_full_call_log(df: pd.DataFrame) -> None:
    print("\n── Full call log (all rows) ─────────────────────────────────────────────")
    display_cols = ["metric", "candidate_idx", "example_idx", "call_idx",
                    "score", "example_input"]
    print(df[display_cols].to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def analyze(path: Path) -> None:
    """Run all analysis sections for the scoring matrix at *path*."""
    print(f"\nLoading: {path}\n")

    data = load(path)
    df   = build_df(data)

    print_header(data, path)
    print_per_metric_meta(data)
    print_score_distribution(df)
    print_score_by_candidate(df)
    print_score_by_example(df)
    print_cross_metric_per_example(df)
    print_candidate_progress(df)
    print_full_call_log(df)

    print(f"\nTotal rows in DataFrame: {len(df)}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _find_latest_matrix()
    analyze(path)


if __name__ == "__main__":
    main()
