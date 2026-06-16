"""Aggregate all scoring_matrix_*.json files from oracle_data_dir into
a single oracle training dataset.

Usage
-----
    # Use the default oracle dir (~/.openjiuwen/oracle):
    python build_oracle_dataset.py

    # Explicit dir:
    python build_oracle_dataset.py ~/.openjiuwen/oracle

    # Dry-run (print only, no CSV written):
    python build_oracle_dataset.py --dry-run

Output files (written to oracle_data_dir):
    oracle_labels.csv      — one row per (run × metric), oracle training table
    oracle_cross_eval.csv  — flat cross-eval rows across all runs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.4f}".format)

DEFAULT_ORACLE_DIR = Path("~/.openjiuwen/oracle").expanduser()


# ── Load & scan ───────────────────────────────────────────────────────────────

def find_matrix_files(oracle_dir: Path) -> list[Path]:
    files = sorted(oracle_dir.glob("scoring_matrix_*.json"))
    if not files:
        sys.exit(f"No scoring_matrix_*.json found in {oracle_dir}")
    return files


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Feature extraction ────────────────────────────────────────────────────────

def _discrimination(calls: list) -> dict:
    """Compute score spread statistics across all calls for one metric run."""
    scores = [c["score"] for c in calls if c.get("score") is not None]
    if not scores:
        return {"discrimination_mean": None, "discrimination_std": None, "discrimination_cv": None}
    m = mean(scores)
    s = stdev(scores) if len(scores) >= 2 else 0.0
    return {
        "discrimination_mean": m,
        "discrimination_std":  s,
        "discrimination_cv":   s / m if m else None,
    }


def _cross_metric_corr(cross_eval: list, metric: str, all_metrics: list) -> Optional[float]:
    """Mean Pearson correlation of *metric* with every other metric, over cross_eval rows."""
    if len(all_metrics) < 2 or not cross_eval:
        return None

    score_cols = [f"score_{m}" for m in all_metrics]
    rows = []
    for row in cross_eval:
        scores = row.get("scores") or {}
        entry = {f"score_{m}": scores.get(m) for m in all_metrics}
        rows.append(entry)

    df = pd.DataFrame(rows).dropna()
    if df.empty or len(df) < 3:
        return None

    # Only keep columns that exist and have variance
    available = [c for c in score_cols if c in df.columns and df[c].std() > 0]
    if f"score_{metric}" not in available or len(available) < 2:
        return None

    corr = df[available].corr()
    col = f"score_{metric}"
    others = [c for c in available if c != col]
    corr_vals = [corr.loc[col, o] for o in others if not pd.isna(corr.loc[col, o])]
    return mean(corr_vals) if corr_vals else None


# ── Build oracle_labels table ─────────────────────────────────────────────────

def build_oracle_labels(files: list[Path]) -> pd.DataFrame:
    """One row per (run_id, metric). Oracle training table.

    Features
    --------
    skill_name, description, n_examples_trainset, n_examples_valset,
    n_examples_holdout, baseline_skill_chars, baseline_skill_body_chars,
    model, metric, baseline_score,
    discrimination_mean, discrimination_std, discrimination_cv,
    n_candidates, n_examples_per_candidate,
    cross_metric_corr_mean

    Labels
    ------
    evolved_score, improvement, accepted,
    rank (1=best metric for this run), is_best_metric
    """
    rows = []

    for path in files:
        try:
            data = load(path)
        except Exception as exc:
            print(f"  [WARN] Cannot load {path.name}: {exc}")
            continue

        run_id     = data.get("run_id", path.stem)
        skill_name = data.get("skill_name", "")
        model      = data.get("model", "")
        sm         = data.get("skill_metadata") or {}
        matrix     = data.get("matrix") or {}
        cross_eval = data.get("cross_eval") or []
        all_metrics = data.get("fitness_metrics") or list(matrix.keys())

        for metric, md in matrix.items():
            if md.get("error"):
                continue  # skip failed metric runs

            disc = _discrimination(md.get("calls") or [])
            corr = _cross_metric_corr(cross_eval, metric, all_metrics)

            rows.append({
                # Identity
                "run_id":          run_id,
                "source_file":     path.name,
                # Skill features
                "skill_name":      skill_name,
                "description":     sm.get("description", ""),
                "n_examples_trainset":    sm.get("n_examples_trainset"),
                "n_examples_valset":      sm.get("n_examples_valset"),
                "n_examples_holdout":     sm.get("n_examples_holdout"),
                "baseline_skill_chars":   sm.get("baseline_skill_chars"),
                "baseline_skill_body_chars": sm.get("baseline_skill_body_chars"),
                "model":           model,
                # Metric identity
                "metric":          metric,
                # Training-time metric signal (features)
                "baseline_score":         md.get("baseline_score"),
                "discrimination_mean":    disc["discrimination_mean"],
                "discrimination_std":     disc["discrimination_std"],
                "discrimination_cv":      disc["discrimination_cv"],
                "n_candidates":           md.get("n_candidates"),
                "n_examples_per_candidate": md.get("n_examples_per_candidate"),
                "cross_metric_corr_mean": corr,
                # Oracle labels
                "evolved_score":   md.get("evolved_score"),
                "improvement":     md.get("improvement"),
                "accepted":        md.get("accepted"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Compute rank and is_best_metric within each run
    df["rank"] = (
        df.groupby("run_id")["improvement"]
          .rank(method="min", ascending=False)
          .astype("Int64")
    )
    df["is_best_metric"] = df["rank"] == 1

    return df


# ── Build oracle_cross_eval table ─────────────────────────────────────────────

def build_oracle_cross_eval(files: list[Path]) -> pd.DataFrame:
    """Flat table of all cross_eval rows across all runs.

    Each row is one unique (run_id, example_id, candidate_output) triple
    scored by all metrics. Used for learning score-correlation patterns.
    """
    rows = []

    for path in files:
        try:
            data = load(path)
        except Exception:
            continue

        run_id     = data.get("run_id", path.stem)
        skill_name = data.get("skill_name", "")
        all_metrics = data.get("fitness_metrics") or []

        for row in data.get("cross_eval") or []:
            entry = {
                "run_id":           run_id,
                "skill_name":       skill_name,
                "example_id":       row.get("example_id", ""),
                "source_metric":    row.get("source_metric", ""),
                "candidate_idx":    row.get("candidate_idx"),
                "gepa_iteration":   row.get("gepa_iteration"),
                "example_input":    (row.get("example_input") or "")[:120],
            }
            scores = row.get("scores") or {}
            for m in all_metrics:
                entry[f"score_{m}"] = scores.get(m)
            rows.append(entry)

    return pd.DataFrame(rows)


# ── Print summary ─────────────────────────────────────────────────────────────

def print_summary(labels: pd.DataFrame, cross_eval: pd.DataFrame, oracle_dir: Path) -> None:
    n_runs   = labels["run_id"].nunique()
    n_skills = labels["skill_name"].nunique()
    n_rows   = len(labels)

    print("=" * 80)
    print("  Oracle Dataset Summary")
    print(f"  Oracle dir   : {oracle_dir}")
    print(f"  Runs         : {n_runs}  |  Skills : {n_skills}  |  Label rows : {n_rows}")
    print(f"  Cross-eval   : {len(cross_eval)} rows")
    print("=" * 80)

    if labels.empty:
        print("  (empty — no valid matrix files found)")
        return

    print("\n── Label table columns ───────────────────────────────────────────────")
    print(f"  {list(labels.columns)}")

    print("\n── Improvement by metric (across all runs) ───────────────────────────")
    agg = (labels.groupby("metric")["improvement"]
                 .agg(["count", "mean", "std", "min", "max"])
                 .rename(columns={"count": "n_runs"}))
    print(agg.to_string())

    print("\n── Best metric frequency (how often each metric wins) ────────────────")
    best = labels[labels["is_best_metric"]].groupby("metric").size()
    print(best.to_string())

    print("\n── Per-skill best metric ─────────────────────────────────────────────")
    best_rows = labels[labels["is_best_metric"]][["skill_name", "metric", "improvement"]].copy()
    print(best_rows.to_string(index=False))

    score_cols = [c for c in cross_eval.columns if c.startswith("score_")]
    if len(score_cols) >= 2 and not cross_eval.empty:
        print("\n── Metric correlation (cross_eval, pooled across all runs) ──────────")
        print(cross_eval[score_cols].corr().to_string())


# ── Main ─────────────────────────────────────────────────────────────────────

def build(oracle_dir: Path, dry_run: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all matrix files and return (oracle_labels_df, cross_eval_df)."""
    print(f"\nScanning: {oracle_dir}")
    files = find_matrix_files(oracle_dir)
    print(f"Found {len(files)} matrix file(s):\n  " + "\n  ".join(f.name for f in files))

    labels    = build_oracle_labels(files)
    cross_eval = build_oracle_cross_eval(files)

    print_summary(labels, cross_eval, oracle_dir)

    if not dry_run and not labels.empty:
        labels_path    = oracle_dir / "oracle_labels.csv"
        cross_eval_path = oracle_dir / "oracle_cross_eval.csv"
        labels.to_csv(labels_path, index=False)
        cross_eval.to_csv(cross_eval_path, index=False)
        print(f"\n  Saved: {labels_path}")
        print(f"  Saved: {cross_eval_path}")

    return labels, cross_eval


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dry_run = "--dry-run" in sys.argv

    oracle_dir = Path(args[0]).expanduser() if args else DEFAULT_ORACLE_DIR
    if not oracle_dir.exists():
        sys.exit(f"Oracle dir not found: {oracle_dir}\n"
                 f"Run gepa_scoring_matrix mode first with oracle_data_dir set in config.json.")

    build(oracle_dir, dry_run=dry_run)


if __name__ == "__main__":
    main()
