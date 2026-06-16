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
import sys
from pathlib import Path

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.fitness_metrics_oracle.oracle_trainer import \
    train

# ── CLI ────────────────────────────────────────────────────────────────────────
DEFAULT_ORACLE_DIR = Path("~/.openjiuwen/oracle").expanduser()


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

    train(data_dir=data_dir, output_dir=output_dir, min_runs=args.min_runs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
