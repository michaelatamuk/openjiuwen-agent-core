#!/usr/bin/env python3
# coding: utf-8
"""
demo_benchmarks.py — Unified benchmark demo for skill_recommender

Downloads any combination of the five benchmark loaders, builds a joint
scoring matrix, and runs sample queries so you can see how the recommender
routes each prompt to the correct skill.

Benchmarks available
--------------------
  bbh        BIG-Bench Hard (23 reasoning tasks) — OPRO, DSPy, TextGrad
  gsm8k      Grade School Math 8K                — OPRO, DSPy
  hotpotqa   HotPotQA multi-hop QA               — DSPy
  pubmedqa   PubMedQA biomedical yes/no/maybe    — SkillGen (2026)
  aquarat    AQuA-RAT algebra word problems       — OPRO

Requirements
------------
    pip install datasets

Run
---
    # All five benchmarks (default)
    python demo_benchmarks.py

    # Specific subset
    python demo_benchmarks.py --benchmarks gsm8k hotpotqa aquarat

    # Persist oracle dir so second run skips downloading
    python demo_benchmarks.py --oracle-dir /tmp/bench_oracle
    python demo_benchmarks.py --oracle-dir /tmp/bench_oracle   # instant
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
from pathlib import Path

# ── path bootstrap ─────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_PKG = "examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender"

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.bbh_loader import load_bbh_to_oracle, DEFAULT_TASKS as BBH_DEFAULT_TASKS  # noqa: E402
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.gsm8k_loader import load_gsm8k_to_oracle     # noqa: E402
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.hotpotqa_loader import load_hotpotqa_to_oracle   # noqa: E402
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.pubmedqa_loader import load_pubmedqa_to_oracle   # noqa: E402
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.aquarat_loader import load_aquarat_to_oracle    # noqa: E402
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.recommender     import build_recommender         # noqa: E402

SEP  = "─" * 72
SEP2 = "═" * 72

# ── Benchmark registry ─────────────────────────────────────────────────────────

ALL_BENCHMARKS = ["bbh", "gsm8k", "hotpotqa", "pubmedqa", "aquarat"]

_LOADERS = {
    "bbh":      lambda d, n, ow: load_bbh_to_oracle(d, tasks=BBH_DEFAULT_TASKS, n_examples=n, overwrite=ow),
    "gsm8k":    lambda d, n, ow: load_gsm8k_to_oracle(d, n_examples=n, overwrite=ow),
    "hotpotqa": lambda d, n, ow: load_hotpotqa_to_oracle(d, n_examples=n, overwrite=ow),
    "pubmedqa": lambda d, n, ow: load_pubmedqa_to_oracle(d, n_examples=n, overwrite=ow),
    "aquarat":  lambda d, n, ow: load_aquarat_to_oracle(d, n_examples=n, overwrite=ow),
}

# ── Sample queries (one per benchmark domain) ──────────────────────────────────
# Each query is worded to resemble prompts that appear in the corresponding
# benchmark so TF-IDF similarity can route them correctly.

SAMPLE_QUERIES = [
    # gsm8k — arithmetic word problem
    (
        "gsm8k",
        "A baker made 48 cookies and packed them into boxes of 6. "
        "He sold 5 boxes. How many cookies does he have left?",
    ),
    # hotpotqa — multi-hop entity reasoning
    (
        "hotpotqa",
        "Who was the lead singer of the band that performed the theme song "
        "for the 1995 James Bond film?",
    ),
    # pubmedqa — biomedical yes/no
    (
        "pubmedqa",
        "Does regular physical exercise reduce the risk of type 2 diabetes "
        "in adults with pre-diabetic conditions?",
    ),
    # aquarat — algebra multiple choice
    (
        "aquarat",
        "If the price of a book is increased by 20% and then decreased by 10%, "
        "what is the net percentage change in the price? "
        "Options:\nA) 8% increase\nB) 10% decrease\nC) 8% decrease\nD) 10% increase\nE) No change",
    ),
    # bbh — logical ordering / word-sorting
    (
        "bbh",
        "Sort the following words in alphabetical order: "
        "zebra mango apple pineapple cherry",
    ),
]


# ── Output helpers ─────────────────────────────────────────────────────────────

def _print_results(
    results: list[dict],
    query: str,
    expected_skill: str,
    is_hit: bool | None = None,
) -> None:
    short = query if len(query) <= 78 else query[:75] + "…"
    print(f"\n{SEP2}")
    print(f"  Query        : {short}")
    print(f"  Expected     : {expected_skill}")
    print(SEP2)
    if not results:
        print("  No recommendations above the thresholds.")
        print(SEP)
        return
    top = results[0]
    if is_hit is None:
        is_hit = (top["skill"] == expected_skill)
    hit = "✓" if is_hit else "✗"
    for i, r in enumerate(results, 1):
        marker = "→" if i == 1 else " "
        print(f"  {marker} [{i}] skill={r['skill']}  "
              f"metric={r['metric']}  score={r['score']:.3f}  "
              f"sim={r['mean_similarity']:.3f}  n={r['n_examples']}")
    print(f"\n  Top-1 match: {hit}  ({top['skill']})")
    print(f"\n{SEP}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skill Recommender — multi-benchmark demo"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=ALL_BENCHMARKS,
        default=ALL_BENCHMARKS,
        metavar="NAME",
        help=f"Benchmarks to load (default: all). Choices: {ALL_BENCHMARKS}",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=30,
        metavar="N",
        help="Examples per benchmark to download (default: 30).",
    )
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Persistent oracle directory. Omit to use a temp dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if JSON files already exist.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.08,
        metavar="F",
        help="Min cosine similarity to include an example (default: 0.08).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.08,
        metavar="F",
        help="Min weighted score to show a result (default: 0.08).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        metavar="N",
        help="Max results per query (default: 3).",
    )
    args = parser.parse_args()

    # ── 1. Oracle directory ───────────────────────────────────────────────────
    ctx: contextlib.AbstractContextManager
    if args.oracle_dir:
        oracle_dir = Path(args.oracle_dir).expanduser()
        ctx = contextlib.nullcontext(oracle_dir)
    else:
        ctx = tempfile.TemporaryDirectory(prefix="benchmarks_oracle_")  # type: ignore[assignment]

    with ctx as tmpdir:
        oracle_dir = Path(tmpdir)

        # ── 2. Download benchmarks ────────────────────────────────────────────
        print(f"\nLoading benchmarks → {oracle_dir}")
        for name in args.benchmarks:
            _LOADERS[name](oracle_dir, args.n_examples, args.overwrite)

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
        loaded_skills = set(rec.skills)
        correct = 0
        total   = 0

        for expected_skill, query in SAMPLE_QUERIES:
            # Resolve expected_skill to the actual skill name in the matrix
            # (bbh produces multiple skills like "word-sorting", etc.)
            if expected_skill == "bbh":
                # any bbh skill counts as a hit
                results = rec.recommend(
                    query           = query,
                    sim_threshold   = args.sim_threshold,
                    score_threshold = args.score_threshold,
                    top_k           = args.top_k,
                )
                bbh_hit = bool(results and results[0]["skill"] in loaded_skills)
                _print_results(results, query, "<any bbh task>", is_hit=bbh_hit)
                if bbh_hit:
                    correct += 1
            else:
                # map benchmark name → skill_name field used in JSON
                skill_map = {
                    "gsm8k":    "math-word-problems",
                    "hotpotqa": "multi-hop-qa",
                    "pubmedqa": "biomedical-qa",
                    "aquarat":  "algebra-reasoning",
                }
                target = skill_map.get(expected_skill, expected_skill)
                if target not in loaded_skills:
                    continue  # benchmark not loaded
                results = rec.recommend(
                    query           = query,
                    sim_threshold   = args.sim_threshold,
                    score_threshold = args.score_threshold,
                    top_k           = args.top_k,
                )
                _print_results(results, query, target)
                if results and results[0]["skill"] == target:
                    correct += 1
            total += 1

        print(f"\n  Routing accuracy: {correct}/{total} queries routed to expected skill")


if __name__ == "__main__":
    main()
