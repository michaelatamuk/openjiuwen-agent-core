from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from .data_features import NUM_COLS, TARGET, GROUP_COL, CAT_MODEL, CAT_METRIC


def _print_report(df: pd.DataFrame, cv_stats: dict, pipeline: Pipeline) -> None:
    sep = "═" * 70
    print(f"\n{sep}")
    print("  Oracle Training Report")
    print(sep)
    print(f"  Training rows  : {len(df)}")
    print(f"  Unique skills  : {df[GROUP_COL].nunique()}")
    print(f"  Unique runs    : {df['run_id'].nunique()}")
    metrics = sorted(df["metric"].unique())
    print(f"  Metrics        : {metrics}")
    print(f"  Target mean    : {df[TARGET].mean():.4f}  std: {df[TARGET].std():.4f}")

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

        metric_ohe_names = list(prep.named_transformers_["metric_ohe"].get_feature_names_out(CAT_METRIC))
        model_ohe_names  = list(prep.named_transformers_["model_ohe"].get_feature_names_out(CAT_MODEL))
        n_svd            = prep.named_transformers_["desc_text"].named_steps["svd"].n_components
        svd_names        = [f"desc_svd_{i}" for i in range(n_svd)]

        feature_names = NUM_COLS + metric_ohe_names + model_ohe_names + svd_names
        importances   = gbr.feature_importances_
        top_n         = min(10, len(feature_names))
        top_idx       = np.argsort(importances)[::-1][:top_n]

        print()
        print(f"── Top {top_n} feature importances ──")
        for i in top_idx:
            print(f"  {feature_names[i]:<40s} {importances[i]:.4f}")
    except Exception:
        pass  # non-critical diagnostic
