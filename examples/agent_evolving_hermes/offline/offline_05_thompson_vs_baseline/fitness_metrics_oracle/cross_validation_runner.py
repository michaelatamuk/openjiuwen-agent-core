from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data_features import TARGET, GROUP_COL, ALL_FEATURE_COLS
from sklearn.model_selection import GroupKFold

from .pipeline_builder import _build_pipeline


# ── Cross-validation ───────────────────────────────────────────────────────────

def _run_cross_validation(df: pd.DataFrame) -> dict:
    """Leave-one-skill-out GroupKFold CV.

    For each fold, fits on all other skills and predicts the held-out skill.
    Evaluation is at the *run* level: does the oracle correctly identify the
    metric with the highest improvement?
    """
    groups   = df[GROUP_COL].values
    n_skills = int(df[GROUP_COL].nunique())
    n_splits = min(5, n_skills)

    if n_splits < 2:
        print(
            "  [WARN] Only 1 unique skill — CV requires ≥ 2 skills (leave-one-out).\n"
            "         Run more scenarios to get meaningful CV results."
        )
        return {}

    gkf = GroupKFold(n_splits=n_splits)
    X = df[ALL_FEATURE_COLS].copy()
    y = df[TARGET].values

    predicted_improvements: list[float] = []
    true_improvements: list[float] = []
    top1_correct: list[bool] = []
    rank_of_true_best: list[int] = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y[train_idx]

        fold_pipe = _build_pipeline(len(train_idx))  # fresh, unfitted copy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_pipe.fit(X_tr, y_tr)
            y_pred = fold_pipe.predict(X_te)

        predicted_improvements.extend(y_pred.tolist())
        true_improvements.extend(y[test_idx].tolist())

        # Per-run evaluation in this fold
        test_df = df.iloc[test_idx].copy()
        test_df["_pred"] = y_pred

        for _run_id, run_df in test_df.groupby("run_id"):
            sorted_by_pred = run_df.sort_values("_pred", ascending=False)
            pred_best = sorted_by_pred.iloc[0]["metric"]
            true_best = run_df.loc[run_df[TARGET].idxmax(), "metric"]
            top1_correct.append(pred_best == true_best)

            metrics_in_run = list(sorted_by_pred["metric"])
            rank = (metrics_in_run.index(true_best) + 1) \
                if true_best in metrics_in_run else len(metrics_in_run)
            rank_of_true_best.append(rank)

    top1_acc  = float(np.mean(top1_correct))  if top1_correct  else float("nan")
    mean_rank = float(np.mean(rank_of_true_best)) if rank_of_true_best else float("nan")
    pearsonr  = (float(np.corrcoef(true_improvements, predicted_improvements)[0, 1])
                 if len(true_improvements) > 1
                 else float("nan"))

    return {"top1_accuracy":        top1_acc,
            "mean_rank_of_best":    mean_rank,
            "improvement_pearsonr": pearsonr,
            "n_runs_evaluated":     len(top1_correct),
            "n_cv_folds":           n_splits}
