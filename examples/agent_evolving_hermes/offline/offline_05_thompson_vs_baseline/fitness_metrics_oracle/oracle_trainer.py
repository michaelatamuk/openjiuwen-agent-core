from __future__ import annotations

import warnings
from pathlib import Path

from .cross_validation_runner import _run_cross_validation
from .data_preparator import _prepare as _prepare_dataset_for_train
from .data_features import TARGET, GROUP_COL, ALL_FEATURE_COLS
from .oracle_predictor import OraclePredictor
from .oracle_dataset_builder import build as _build_dataset_for_train
from .pipeline_builder import _build_pipeline
from .report_printer import _print_report


# ── Public training entry-point ────────────────────────────────────────────────

def train(data_dir: Path, output_dir: Path, min_runs: int = 3, dry_run: bool = False) -> OraclePredictor | None:
    """Aggregate data, train, evaluate and save the oracle.

    Returns the fitted OraclePredictor, or None if dry_run=True.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — aggregate scoring_matrix_*.json → oracle_labels.csv
    labels_df, _ = _build_dataset_for_train(data_dir, dry_run=False)

    if labels_df.empty:
        print("[ERROR] No valid oracle label data found. Run gepa_scoring_matrix mode first.")
        return None

    # Step 2 — prepare features / filter bad rows
    dataset_as_df = _prepare_dataset_for_train(labels_df)

    if dataset_as_df.empty:
        print("[ERROR] All rows were filtered out (no valid improvement labels).")
        return None

    n_runs   = int(dataset_as_df["run_id"].nunique())
    n_skills = int(dataset_as_df[GROUP_COL].nunique())

    print(f"\n{'═' * 70}")
    print(f"  Oracle Trainer  —  {n_runs} run(s), {n_skills} skill(s), {len(dataset_as_df)} rows")

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
    train_pipeline = _build_pipeline(n_runs)

    # Step 4 — cross-validation
    cv_stats = _run_cross_validation(dataset_as_df)

    # Step 5 — fit on all data
    X_all = dataset_as_df[ALL_FEATURE_COLS].copy()
    y_all = dataset_as_df[TARGET].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_pipeline.fit(X_all, y_all)

    # Step 6 — report
    _print_report(dataset_as_df, cv_stats, train_pipeline)

    # Step 7 — package and save
    predictor = OraclePredictor(pipeline=train_pipeline,
                                metrics_seen=sorted(dataset_as_df["metric"].unique().tolist()),
                                feature_meta={
            "n_runs":    n_runs,
            "n_skills":  n_skills,
            "data_dir":  str(data_dir),
            "cv":        cv_stats,
        })
    predictor.save(output_dir / "oracle_model.pkl")
    return predictor
