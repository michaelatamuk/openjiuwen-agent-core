#!/usr/bin/env python3
# coding: utf-8
"""
train_oracle.py — Fitness Metrics Oracle Trainer

Reads accumulated scoring_matrix_*.json files from oracle_data_dir,
trains a scikit-learn model that predicts which fitness metric will
yield the best improvement for a given skill (a-priori — before any GEPA run).

Usage
-----
    python train_oracle.py [OPTIONS]

    --data-dir PATH     Oracle data directory  [default: ~/.openjiuwen/oracle]
    --output PATH       Where to save oracle_model.pkl  [default: <data-dir>]
    --min-runs N        Warn if fewer than N runs available  [default: 3]
    --dry-run           Build CSVs and print stats only; skip training

Output
------
    <output>/oracle_model.pkl   Serialised OraclePredictor

Example (inference after training)
-----------------------------------
    from train_oracle import OraclePredictor

    pred = OraclePredictor.load("~/.openjiuwen/oracle/oracle_model.pkl")
    ranking = pred.predict(
        skill_metadata={
            "description": "Answers customer support questions about SmartHub devices.",
            "n_examples_trainset": 20,
            "n_examples_valset": 5,
            "n_examples_holdout": 10,
            "baseline_skill_chars": 3400,
            "baseline_skill_body_chars": 2800,
        },
        candidate_metrics=["bag_of_words", "f1", "graph"],
    )
    print(ranking[0]["metric"])   # predicted best metric
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── scikit-learn import guard ──────────────────────────────────────────────────

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as _err:
    sys.exit(
        f"scikit-learn is required but not installed:\n  {_err}\n\n"
        f"Install it with:  pip install scikit-learn"
    )

# ── sibling import (build_oracle_dataset lives in the same directory) ──────────

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from build_oracle_dataset import build as _aggregate  # noqa: E402

DEFAULT_ORACLE_DIR = Path("~/.openjiuwen/oracle").expanduser()

# ── Feature definition ─────────────────────────────────────────────────────────

# Only columns available BEFORE any GEPA run (a-priori).
# Discrimination stats (discrimination_mean / cv / ...) and cross_metric_corr_mean
# are computed from the GEPA call log and are therefore excluded.
_NUM_COLS = [
    "n_examples_trainset",
    "n_examples_valset",
    "n_examples_holdout",
    "baseline_skill_chars",
    "baseline_skill_body_chars",
]
_CAT_METRIC = ["metric"]
_CAT_MODEL  = ["model"]
_TEXT_COL   = "description"
_TARGET     = "improvement"
_GROUP_COL  = "skill_name"

_ALL_FEATURE_COLS = _NUM_COLS + _CAT_METRIC + _CAT_MODEL + [_TEXT_COL]


# ── OraclePredictor ────────────────────────────────────────────────────────────

class OraclePredictor:
    """Wraps a trained sklearn Pipeline and exposes a simple predict() interface.

    Intended to be pickled with ``save()`` and restored with ``load()``.
    """

    def __init__(
        self,
        pipeline: Any,
        metrics_seen: list[str],
        feature_meta: dict,
    ) -> None:
        self._pipeline     = pipeline
        self._metrics_seen = metrics_seen
        self._feature_meta = feature_meta  # diagnostics / provenance

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(
        self,
        skill_metadata: dict,
        candidate_metrics: list[str],
        model: str = "",
    ) -> list[dict]:
        """Rank candidate_metrics by predicted improvement (best first).

        Parameters
        ----------
        skill_metadata : dict
            A-priori skill features. Recognised keys:
              description, n_examples_trainset, n_examples_valset,
              n_examples_holdout, baseline_skill_chars, baseline_skill_body_chars
        candidate_metrics : list[str]
            Metric names to rank (e.g. ["bag_of_words", "f1", "graph"]).
        model : str
            LLM model string (e.g. "deepseek/deepseek-chat").
            Use "" or omit if unknown — the model will fall back to the
            closest seen value via OneHotEncoder's handle_unknown="ignore".

        Returns
        -------
        list[dict]
            Sorted best-first:
            [{"metric": str, "predicted_improvement": float}, ...]
        """
        rows = [
            {
                "description":               skill_metadata.get("description", ""),
                "n_examples_trainset":       skill_metadata.get("n_examples_trainset") or 0,
                "n_examples_valset":         skill_metadata.get("n_examples_valset") or 0,
                "n_examples_holdout":        skill_metadata.get("n_examples_holdout") or 0,
                "baseline_skill_chars":      skill_metadata.get("baseline_skill_chars") or 0,
                "baseline_skill_body_chars": skill_metadata.get("baseline_skill_body_chars") or 0,
                "metric": m,
                "model":  model or "unknown",
            }
            for m in candidate_metrics
        ]
        df = pd.DataFrame(rows)
        _fill_numeric(df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self._pipeline.predict(df[_ALL_FEATURE_COLS])

        results = [
            {"metric": m, "predicted_improvement": float(p)}
            for m, p in zip(candidate_metrics, preds)
        ]
        results.sort(key=lambda x: x["predicted_improvement"], reverse=True)
        return results

    @property
    def metrics_seen(self) -> list[str]:
        """Metric names the oracle was trained on."""
        return list(self._metrics_seen)

    @property
    def meta(self) -> dict:
        """Provenance info: n_runs, n_skills, CV stats, etc."""
        return dict(self._feature_meta)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Serialise this predictor to *path* via pickle."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        print(f"  Saved: {path}")

    @classmethod
    def load(cls, path: Path | str) -> "OraclePredictor":
        """Load a previously saved OraclePredictor."""
        path = Path(path).expanduser()
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected OraclePredictor, got {type(obj).__name__}")
        return obj


# ── Data preparation ───────────────────────────────────────────────────────────

def _fill_numeric(df: pd.DataFrame) -> None:
    """Fill missing numeric feature columns with 0 in-place."""
    for col in _NUM_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, feature-ready DataFrame with only rows that have a valid label."""
    df = df.copy()
    df = df[df[_TARGET].notna()].copy()
    _fill_numeric(df)
    df[_TEXT_COL] = df[_TEXT_COL].fillna("").astype(str)
    df["model"]   = df["model"].fillna("unknown").astype(str)
    df["metric"]  = df["metric"].fillna("unknown").astype(str)
    if _GROUP_COL not in df.columns:
        df[_GROUP_COL] = "unknown_skill"
    return df.reset_index(drop=True)


# ── Model construction ─────────────────────────────────────────────────────────

def _build_pipeline(n_runs: int) -> Pipeline:
    """Return an unfitted sklearn Pipeline.

    Uses lighter settings when data is scarce (< 20 runs) to reduce overfitting.
    """
    small        = n_runs < 20
    n_estimators = 30  if small else 100
    max_depth    = 2   if small else 3
    # TruncatedSVD requires n_components < n_samples; use floor(n_runs/2) so
    # that even the smallest training fold (≈half the data) stays above n_svd.
    n_svd = max(1, min(8, n_runs // 2))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                _NUM_COLS,
            ),
            (
                "metric_ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _CAT_METRIC,
            ),
            (
                "model_ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _CAT_MODEL,
            ),
            (
                "desc_text",
                Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=200, ngram_range=(1, 2))),
                    ("svd",   TruncatedSVD(n_components=n_svd, random_state=42)),
                ]),
                _TEXT_COL,  # single column name → passed as 1-D array to TfidfVectorizer
            ),
        ],
        sparse_threshold=0,  # always produce dense output
    )

    return Pipeline([
        ("prep", preprocessor),
        ("gbr",  GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])


# ── Cross-validation ───────────────────────────────────────────────────────────

def _run_cv(df: pd.DataFrame, pipeline: Pipeline) -> dict:
    """Leave-one-skill-out GroupKFold CV.

    For each fold, fits on all other skills and predicts the held-out skill.
    Evaluation is at the *run* level: does the oracle correctly identify the
    metric with the highest improvement?
    """
    groups   = df[_GROUP_COL].values
    n_skills = int(df[_GROUP_COL].nunique())
    n_splits = min(5, n_skills)

    if n_splits < 2:
        print(
            "  [WARN] Only 1 unique skill — CV requires ≥ 2 skills (leave-one-out).\n"
            "         Run more scenarios to get meaningful CV results."
        )
        return {}

    gkf = GroupKFold(n_splits=n_splits)
    X   = df[_ALL_FEATURE_COLS].copy()
    y   = df[_TARGET].values

    predicted_improvements: list[float] = []
    true_improvements:      list[float] = []
    top1_correct:           list[bool]  = []
    rank_of_true_best:      list[int]   = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr        = y[train_idx]

        fold_pipe = _build_pipeline(len(train_idx))  # fresh, unfitted copy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_pipe.fit(X_tr, y_tr)
            y_pred = fold_pipe.predict(X_te)

        predicted_improvements.extend(y_pred.tolist())
        true_improvements.extend(y[test_idx].tolist())

        # Per-run evaluation in this fold
        test_df        = df.iloc[test_idx].copy()
        test_df["_pred"] = y_pred

        for _run_id, run_df in test_df.groupby("run_id"):
            sorted_by_pred   = run_df.sort_values("_pred", ascending=False)
            pred_best        = sorted_by_pred.iloc[0]["metric"]
            true_best        = run_df.loc[run_df[_TARGET].idxmax(), "metric"]
            top1_correct.append(pred_best == true_best)

            metrics_in_run   = list(sorted_by_pred["metric"])
            rank             = (metrics_in_run.index(true_best) + 1) if true_best in metrics_in_run else len(metrics_in_run)
            rank_of_true_best.append(rank)

    top1_acc  = float(np.mean(top1_correct))  if top1_correct  else float("nan")
    mean_rank = float(np.mean(rank_of_true_best)) if rank_of_true_best else float("nan")
    pearsonr  = (
        float(np.corrcoef(true_improvements, predicted_improvements)[0, 1])
        if len(true_improvements) > 1
        else float("nan")
    )

    return {
        "top1_accuracy":        top1_acc,
        "mean_rank_of_best":    mean_rank,
        "improvement_pearsonr": pearsonr,
        "n_runs_evaluated":     len(top1_correct),
        "n_cv_folds":           n_splits,
    }


# ── Report printing ────────────────────────────────────────────────────────────

def _print_report(df: pd.DataFrame, cv_stats: dict, pipeline: Pipeline) -> None:
    sep = "═" * 70
    print(f"\n{sep}")
    print("  Oracle Training Report")
    print(sep)
    print(f"  Training rows  : {len(df)}")
    print(f"  Unique skills  : {df[_GROUP_COL].nunique()}")
    print(f"  Unique runs    : {df['run_id'].nunique()}")
    metrics = sorted(df["metric"].unique())
    print(f"  Metrics        : {metrics}")
    print(f"  Target mean    : {df[_TARGET].mean():.4f}  std: {df[_TARGET].std():.4f}")

    if cv_stats:
        print()
        print(f"── Leave-one-skill-out CV  ({cv_stats['n_cv_folds']} folds) ──")
        top1 = cv_stats["top1_accuracy"]
        rank = cv_stats["mean_rank_of_best"]
        corr = cv_stats["improvement_pearsonr"]
        n_ev = cv_stats["n_runs_evaluated"]
        print(f"  Top-1 accuracy          : {top1:.1%}  ({n_ev} runs evaluated)")
        print(f"  Mean rank of true best  : {rank:.2f}   (1.0 = always correct)")
        print(f"  Improvement Pearson r   : {corr:.3f}")

    # Per-metric best-metric frequency
    print()
    print("── Best metric frequency (training data) ──")
    freq = df[df["is_best_metric"]].groupby("metric").size().sort_values(ascending=False)
    for m, cnt in freq.items():
        pct = cnt / df["run_id"].nunique()
        print(f"  {m:<25s}  wins {cnt:2d} / {df['run_id'].nunique()} runs  ({pct:.0%})")

    # Feature importances
    try:
        gbr  = pipeline.named_steps["gbr"]
        prep = pipeline.named_steps["prep"]

        metric_ohe_names = list(prep.named_transformers_["metric_ohe"].get_feature_names_out(_CAT_METRIC))
        model_ohe_names  = list(prep.named_transformers_["model_ohe"].get_feature_names_out(_CAT_MODEL))
        n_svd            = prep.named_transformers_["desc_text"].named_steps["svd"].n_components
        svd_names        = [f"desc_svd_{i}" for i in range(n_svd)]

        feature_names = _NUM_COLS + metric_ohe_names + model_ohe_names + svd_names
        importances   = gbr.feature_importances_
        top_n         = min(10, len(feature_names))
        top_idx       = np.argsort(importances)[::-1][:top_n]

        print()
        print(f"── Top {top_n} feature importances ──")
        for i in top_idx:
            print(f"  {feature_names[i]:<40s} {importances[i]:.4f}")
    except Exception:
        pass  # non-critical diagnostic


# ── Public training entry-point ────────────────────────────────────────────────

def train(
    data_dir: Path,
    output_dir: Path,
    min_runs: int = 3,
    dry_run: bool = False,
) -> OraclePredictor | None:
    """Aggregate data, train, evaluate and save the oracle.

    Returns the fitted OraclePredictor, or None if dry_run=True.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — aggregate scoring_matrix_*.json → oracle_labels.csv
    labels_df, _ = _aggregate(data_dir, dry_run=False)

    if labels_df.empty:
        print("[ERROR] No valid oracle label data found. Run gepa_scoring_matrix mode first.")
        return None

    # Step 2 — prepare features / filter bad rows
    df = _prepare(labels_df)

    if df.empty:
        print("[ERROR] All rows were filtered out (no valid improvement labels).")
        return None

    n_runs   = int(df["run_id"].nunique())
    n_skills = int(df[_GROUP_COL].nunique())

    print(f"\n{'═' * 70}")
    print(f"  Oracle Trainer  —  {n_runs} run(s), {n_skills} skill(s), {len(df)} rows")

    if n_runs < min_runs:
        print(
            f"\n  [WARN] Only {n_runs} run(s) available (min_runs={min_runs}).\n"
            f"         Model will be trained, but predictions have very low confidence.\n"
            f"         Run more gepa_scoring_matrix passes across diverse skills to improve quality."
        )

    if dry_run:
        print("\n  [dry-run] Skipping model training.")
        return None

    # Step 3 — build pipeline
    pipeline = _build_pipeline(n_runs)

    # Step 4 — cross-validation
    cv_stats = _run_cv(df, pipeline)

    # Step 5 — fit on all data
    X_all = df[_ALL_FEATURE_COLS].copy()
    y_all = df[_TARGET].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_all, y_all)

    # Step 6 — report
    _print_report(df, cv_stats, pipeline)

    # Step 7 — package and save
    predictor = OraclePredictor(
        pipeline=pipeline,
        metrics_seen=sorted(df["metric"].unique().tolist()),
        feature_meta={
            "n_runs":    n_runs,
            "n_skills":  n_skills,
            "data_dir":  str(data_dir),
            "cv":        cv_stats,
        },
    )
    predictor.save(output_dir / "oracle_model.pkl")
    return predictor


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train_oracle.py",
        description="Train a fitness metrics oracle from accumulated scoring matrix data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python train_oracle.py --data-dir ~/.openjiuwen/oracle\n\n"
            "After training, use the predictor in Python:\n"
            "  from train_oracle import OraclePredictor\n"
            "  p = OraclePredictor.load('~/.openjiuwen/oracle/oracle_model.pkl')\n"
            "  print(p.predict({'description': '...', 'n_examples_trainset': 20, ...},\n"
            "                  ['bag_of_words', 'f1', 'graph']))\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        type=Path,
        default=DEFAULT_ORACLE_DIR,
        help="Directory with scoring_matrix_*.json files  [default: ~/.openjiuwen/oracle]",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        default=None,
        help="Where to write oracle_model.pkl  [default: same as --data-dir]",
    )
    parser.add_argument(
        "--min-runs",
        metavar="N",
        type=int,
        default=3,
        help="Warn if fewer than N GEPA runs are available  [default: 3]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Aggregate data and print stats only; do not train or save",
    )

    args       = parser.parse_args()
    data_dir   = args.data_dir.expanduser()
    output_dir = (args.output or args.data_dir).expanduser()

    if not data_dir.exists():
        sys.exit(
            f"Oracle data directory not found: {data_dir}\n"
            f"Run gepa_scoring_matrix mode first with oracle_data_dir set in config.json."
        )

    train(
        data_dir=data_dir,
        output_dir=output_dir,
        min_runs=args.min_runs,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
