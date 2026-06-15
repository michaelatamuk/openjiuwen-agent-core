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
                "metric":           metric_name,
                "candidate_idx":    call.get("candidate_idx"),
                "example_idx":      call.get("example_idx"),
                "example_id":       call.get("example_id", ""),
                "gepa_iteration":   call.get("gepa_iteration"),
                "call_idx":         call.get("call_idx"),
                "score":            call.get("score"),
                "example_input":    _truncate(call.get("example_input", ""), 70),
                "candidate_output": _truncate(call.get("candidate_output", ""), 70),
            })
    return pd.DataFrame(rows)


def build_cross_eval_df(data: dict) -> pd.DataFrame:
    """Build a DataFrame from the cross_eval section."""
    rows = []
    for row in data.get("cross_eval", []):
        base = {
            "example_id":       row.get("example_id", ""),
            "source_metric":    row.get("source_metric", ""),
            "candidate_idx":    row.get("candidate_idx"),
            "gepa_iteration":   row.get("gepa_iteration"),
            "example_input":    _truncate(row.get("example_input", ""), 60),
            "candidate_output": _truncate(row.get("candidate_output", ""), 60),
        }
        for metric_name, score in (row.get("scores") or {}).items():
            base[f"score_{metric_name}"] = score
        rows.append(base)
    return pd.DataFrame(rows)


# ── Print sections ────────────────────────────────────────────────────────────

def print_header(data: dict, path: Path) -> None:
    s  = data["summary"]
    sm = data.get("skill_metadata") or {}
    print("=" * 90)
    print(f"  Scoring Matrix Analysis")
    print(f"  Run ID  : {data.get('run_id', 'n/a')}")
    print(f"  Skill   : {data['skill_name']}")
    if sm.get("description"):
        print(f"  Desc    : {_truncate(sm['description'], 80)}")
    if sm:
        n_train   = sm.get("n_examples_trainset", "?")
        n_val     = sm.get("n_examples_valset", "?")
        n_holdout = sm.get("n_examples_holdout")
        holdout_s = f"  holdout={n_holdout}" if n_holdout is not None else ""
        print(f"  Dataset : train={n_train}  val={n_val}{holdout_s}")
        b_chars = sm.get("baseline_skill_chars")
        body_chars = sm.get("baseline_skill_body_chars")
        if b_chars is not None:
            print(f"  Skill   : {b_chars} chars total  ({body_chars} body)")
        extra = sm.get("frontmatter_extra") or {}
        if extra:
            print(f"  FM extra: {extra}")
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
        baseline_text = mv.get("baseline_skill_text") or ""
        evolved_text  = mv.get("evolved_skill_text")  or ""
        rows.append({
            "metric":                 mn,
            "n_candidates":           mv.get("n_candidates"),
            "examples_per_candidate": mv.get("n_examples_per_candidate"),
            "total_calls":            len(mv.get("calls", [])),
            "baseline_score":         mv.get("baseline_score"),
            "evolved_score":          mv.get("evolved_score"),
            "improvement":            mv.get("improvement"),
            "accepted":               mv.get("accepted"),
            "baseline_skill_chars":   len(baseline_text),
            "evolved_skill_chars":    len(evolved_text),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # Print skill text excerpts for the first metric where they exist
    for mn, mv in data["matrix"].items():
        bst = mv.get("baseline_skill_text")
        est = mv.get("evolved_skill_text")
        if bst:
            print(f"\n  baseline_skill_text [{mn}] (first 200 chars):")
            print(f"    {_truncate(bst, 200)}")
        if est:
            print(f"  evolved_skill_text  [{mn}] (first 200 chars):")
            print(f"    {_truncate(est, 200)}")
        if bst or est:
            break  # only show first metric to keep output readable


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


def _example_key(df: pd.DataFrame) -> str:
    """Return 'example_id' if the column is populated, else fall back to 'example_idx'."""
    if "example_id" in df.columns and df["example_id"].ne("").any():
        return "example_id"
    return "example_idx"


def print_score_by_example(df: pd.DataFrame) -> None:
    key = _example_key(df)
    print(f"\n── Mean score per (metric, {key}) — which examples are harder? ─────────")
    pivot = (df.groupby(["metric", key])["score"]
               .agg(["mean", "std", "count"])
               .rename(columns={"mean": "score_mean", "std": "score_std", "count": "n_evals"}))
    print(pivot.to_string())


def print_cross_metric_per_example(df: pd.DataFrame) -> None:
    key = _example_key(df)
    print(f"\n── Cross-metric score per {key} (mean across all candidates) ──────────")
    pivot = (df.groupby([key, "metric"])["score"]
               .mean()
               .unstack("metric"))
    pivot["row_mean"] = pivot.mean(axis=1)
    print(pivot.to_string())


def print_gepa_iteration_analysis(df: pd.DataFrame) -> None:
    print("\n── Score by gepa_iteration (estimated) ─────────────────────────────────")
    if "gepa_iteration" not in df.columns or df["gepa_iteration"].isna().all():
        print("  (no gepa_iteration data)")
        return
    pivot = (df.groupby(["gepa_iteration", "metric"])["score"]
               .agg(["mean", "count"])
               .rename(columns={"mean": "score_mean", "count": "n_evals"}))
    print(pivot.to_string())


def print_candidate_progress(df: pd.DataFrame) -> None:
    print("\n── Score progression by candidate_idx (did GEPA improve over time?) ────")
    pivot = (df.groupby(["candidate_idx", "metric"])["score"]
               .mean()
               .unstack("metric"))
    pivot["row_mean"] = pivot.mean(axis=1)
    print(pivot.to_string())


def print_cross_eval(data: dict) -> None:
    """Print the cross-evaluation oracle table."""
    cross_eval = data.get("cross_eval", [])
    print(f"\n── Cross-eval oracle table ({len(cross_eval)} unique (example, output) pairs) ─")
    if not cross_eval:
        print("  (no cross_eval data — run with updated gepa_scoring_matrix mode)")
        return

    df = build_cross_eval_df(data)
    score_cols = [c for c in df.columns if c.startswith("score_")]

    # Summary: mean score per metric across all pairs
    if score_cols:
        print("\n  Mean score per metric (all unique pairs):")
        print(f"    {df[score_cols].mean().to_string()}")

    # Per source_metric breakdown: which GEPA run's candidates score how on every metric
    if "source_metric" in df.columns and score_cols:
        print("\n  Mean score by source_metric × evaluated metric:")
        print(df.groupby("source_metric")[score_cols].mean().to_string())

    # Correlation between metrics (using only rows where all scores present)
    if len(score_cols) >= 2:
        corr = df[score_cols].corr()
        print("\n  Metric correlation matrix (on cross-eval pairs):")
        print(corr.to_string())

    # Full table (truncated columns)
    display_cols = ["example_id", "source_metric", "candidate_idx", "gepa_iteration",
                    "example_input"] + score_cols
    display_cols = [c for c in display_cols if c in df.columns]
    print(f"\n  Full cross-eval rows:")
    print(df[display_cols].to_string(index=False))


def print_full_call_log(df: pd.DataFrame) -> None:
    print("\n── Full call log (all rows) ─────────────────────────────────────────────")
    display_cols = ["metric", "candidate_idx", "example_idx", "example_id",
                    "gepa_iteration", "call_idx", "score", "example_input"]
    display_cols = [c for c in display_cols if c in df.columns]
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
    print_gepa_iteration_analysis(df)
    print_candidate_progress(df)
    print_cross_eval(data)
    print_full_call_log(df)

    print(f"\nTotal rows in DataFrame: {len(df)}")
    cross_eval_count = len(data.get("cross_eval", []))
    if cross_eval_count:
        print(f"Cross-eval rows        : {cross_eval_count}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _find_latest_matrix()
    analyze(path)


if __name__ == "__main__":
    main()
