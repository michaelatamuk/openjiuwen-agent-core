#!/usr/bin/env python3
# coding: utf-8
"""
demo_bbh.py — Skill Recommender demo using BIG-Bench Hard

Downloads a subset of the BIG-Bench Hard benchmark from HuggingFace,
builds a scoring matrix from a deterministic baseline (no LLM required),
loads it through the standard skill_recommender stack, and runs several
sample queries so you can see which BBH task (skill) the recommender
surfaces for each prompt type.

Requirements
------------
    pip install datasets   # HuggingFace datasets

Run
---
    # Use default 6 BBH tasks
    python demo_bbh.py

    # Choose specific tasks
    python demo_bbh.py --tasks logical_deduction_three_objects date_understanding word_sorting

    # Persist the oracle dir so subsequent runs skip downloading
    python demo_bbh.py --oracle-dir /tmp/bbh_oracle
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── path bootstrap ─────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.bbh_loader import (  # noqa: E402
    ALL_TASKS,
    DEFAULT_TASKS,
    load_bbh_to_oracle,
)
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.recommender import (  # noqa: E402
    build_recommender,
)

SEP  = "─" * 72
SEP2 = "═" * 72

# ── Sample queries ─────────────────────────────────────────────────────────────
# Each query is phrased to resemble one of the default BBH task types so the
# recommender can demonstrate meaningful routing.

SAMPLE_QUERIES = [
    # logical-deduction
    "Alice is taller than Bob, and Bob is taller than Charlie. "
    "Who is the shortest of the three?",

    # date-understanding
    "If today is the third Monday of April 2023, what date was exactly "
    "two weeks ago?",

    # object-counting
    "There are 4 red balls and 3 blue balls in a bag. "
    "How many balls are there in total?",

    # causal-judgement
    "Would skipping breakfast every morning likely cause decreased "
    "concentration in the afternoon?",

    # movie-recommendation
    "I enjoy cerebral science-fiction thrillers with unexpected endings. "
    "Which movie would you recommend?",

    # word-sorting
    "Sort the following words in alphabetical order: "
    "mango watermelon cherry apple pineapple",
]


# ── Output helpers ─────────────────────────────────────────────────────────────

def _print_results(results: list[dict], query: str) -> None:
    short = query if len(query) <= 80 else query[:77] + "…"
    print(f"\n{SEP2}")
    print(f"  Query : {short}")
    print(SEP2)
    if not results:
        print("  No recommendations above the thresholds.")
        print(SEP)
        return
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Skill  : {r['skill']}")
        print(f"       Metric : {r['metric']}")
        print(f"       Score  : {r['score']:.3f}  "
              f"(mean_sim={r['mean_similarity']:.3f}, "
              f"{r['n_examples']} example(s))")
        for ex in r["similar_examples"]:
            inp = ex.get("input",    "")[:90]
            exp = ex.get("expected", "")[:60]
            out = ex.get("output",   "")[:60]
            print(f"       ↳ [{ex['similarity']:.3f}] prompt   : {inp}")
            if exp:
                print(f"                 expected : {exp}")
            if out:
                print(f"                 output   : {out}")
    print(f"\n{SEP}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skill Recommender — BIG-Bench Hard demo"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK",
        help=f"BBH task names to load (default: {DEFAULT_TASKS}). "
             f"Available: {ALL_TASKS}",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=30,
        metavar="N",
        help="Examples per task to download (default: 30).",
    )
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Persistent directory for scoring matrix JSONs. "
             "If omitted, a temporary directory is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if JSON files already exist.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.10,
        metavar="F",
        help="Min cosine similarity to include an example (default: 0.10).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.10,
        metavar="F",
        help="Min weighted score to show a result (default: 0.10).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        metavar="N",
        help="Max results per query (default: 3).",
    )
    args = parser.parse_args()

    tasks = args.tasks or DEFAULT_TASKS

    # ── 1. Build / reuse oracle dir ───────────────────────────────────────────
    import tempfile, contextlib

    ctx: contextlib.AbstractContextManager
    if args.oracle_dir:
        oracle_dir = Path(args.oracle_dir).expanduser()
        ctx = contextlib.nullcontext(oracle_dir)
    else:
        ctx = tempfile.TemporaryDirectory(prefix="bbh_oracle_")  # type: ignore[assignment]

    with ctx as tmpdir:
        oracle_dir = Path(tmpdir)

        # ── 2. Download BBH tasks ─────────────────────────────────────────────
        print(f"\nDownloading BBH tasks → {oracle_dir}")
        load_bbh_to_oracle(
            oracle_dir  = oracle_dir,
            tasks       = tasks,
            n_examples  = args.n_examples,
            overwrite   = args.overwrite,
        )

        # ── 3. Build recommender ──────────────────────────────────────────────
        print("\nBuilding recommender …")
        rec = build_recommender(
            oracle_dir      = oracle_dir,
            variant         = "baseline",
            embedder_method = "tfidf",
        )
        print(f"  Rows    : {rec.n_examples}")
        print(f"  Skills  : {rec.skills}")
        print(f"  Metrics : {rec.metrics}")

        # ── 4. Run queries ────────────────────────────────────────────────────
        for query in SAMPLE_QUERIES:
            results = rec.recommend(
                query           = query,
                sim_threshold   = args.sim_threshold,
                score_threshold = args.score_threshold,
                top_k           = args.top_k,
            )
            _print_results(results, query)


if __name__ == "__main__":
    main()
