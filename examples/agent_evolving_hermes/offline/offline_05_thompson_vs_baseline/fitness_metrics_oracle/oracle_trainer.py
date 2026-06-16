from __future__ import annotations

import warnings
from pathlib import Path

from .cross_validation_runner import _run_cross_validation
from .data_preparator import _prepare
from .data_features import TARGET, GROUP_COL, ALL_FEATURE_COLS
from .oracle_predictor import OraclePredictor
from .oracle_dataset_builder import build as _aggregate
from .pipeline_builder import _build_pipeline
from .report_printer import _print_report


# ── Public training entry-point ────────────────────────────────────────────────

def train(data_dir: Path, output_dir: Path, min_runs: int = 3, dry_run: bool = False) -> OraclePredictor | None:
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
    n_skills = int(df[GROUP_COL].nunique())

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
    cv_stats = _run_cross_validation(df, pipeline)

    # Step 5 — fit on all data
    X_all = df[ALL_FEATURE_COLS].copy()
    y_all = df[TARGET].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_all, y_all)

    # Step 6 — report
    _print_report(df, cv_stats, pipeline)

    # Step 7 — package and save
    predictor = OraclePredictor(pipeline=pipeline,
                                metrics_seen=sorted(df["metric"].unique().tolist()),
                                feature_meta={
            "n_runs":    n_runs,
            "n_skills":  n_skills,
            "data_dir":  str(data_dir),
            "cv":        cv_stats,
        })
    predictor.save(output_dir / "oracle_model.pkl")
    return predictor
