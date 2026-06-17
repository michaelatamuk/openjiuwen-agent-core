from __future__ import annotations

import argparse
from pathlib import Path

from .runner_benchmarks import ALL_BENCHMARKS


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_ORACLE_DIR  = Path("~/.openjiuwen/oracle").expanduser()



# ── CLI ───────────────────────────────────────────────────────────────────────

def args_parser():
    parser = argparse.ArgumentParser(prog="runner.py",
                                     description="Skill Recommender — query, demo, or benchmark mode.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__,)

    # ── Mode flags ────────────────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--demo",
                            action="store_true",
                            default=False,
                            help="Run the synthetic quick-start demo (no network required).")
    mode_group.add_argument("--benchmarks",
                            nargs="*",
                            metavar="NAME",
                            default=None,
                            help=f"Download HuggingFace benchmarks and show routing accuracy. "
                                 f"Pass names to select a subset, or omit names for all. "
                                 f"Available: {', '.join(ALL_BENCHMARKS)}",)

    # ── Query input (query mode only) ─────────────────────────────────────
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("query",
                             nargs="?",
                             default=None,
                             help="Prompt to find skills for (query mode).  Pass '-' to read from stdin.",)
    query_group.add_argument("--from-file",
                             metavar="PATH",
                             type=Path,
                             default=None,
                             help="File with one query per line (query mode).")
    query_group.add_argument("--list-skills",
                             action="store_true",
                             default=False,
                             help="List all skills in the matrix and exit (query mode).",)

    # ── Shared settings ───────────────────────────────────────────────────
    parser.add_argument("--data-dir",
                        metavar="PATH",
                        type=Path,
                        default=DEFAULT_ORACLE_DIR,
                        help="Directory with scoring_matrix_*.json files  [default: ~/.openjiuwen/oracle]",)
    parser.add_argument("--oracle-dir",
                        metavar="PATH",
                        type=Path,
                        default=None,
                        help="Alias for --data-dir (accepted in all modes).",)
    parser.add_argument("--variant",
                        choices=["baseline", "evolved", "both"],
                        default="baseline",
                        help="Matrix layer to use  [default: baseline]",)
    parser.add_argument("--embedder",
                        choices=["tfidf", "openai"],
                        default="tfidf",
                        help="Embedding backend  [default: tfidf]",)
    parser.add_argument("--sim-threshold",
                        type=float,
                        default=None,
                        metavar="FLOAT",
                        help="Min cosine similarity  [default: 0.25 for query, 0.08 for benchmarks]",)
    parser.add_argument("--score-threshold",
                        type=float,
                        default=None,
                        metavar="FLOAT",
                        help="Min weighted score  [default: 0.20 for query, 0.08 for benchmarks]",)
    parser.add_argument("--top-k",
                        type=int,
                        default=None,
                        metavar="N",
                        help="Max results per query  [default: 10 for query, 3 for benchmarks]",)

    # ── Query-only settings ───────────────────────────────────────────────
    parser.add_argument("--min-examples",
                        type=int,
                        default=1,
                        metavar="N",
                        help="Min similar examples required (query mode)  [default: 1]",)
    parser.add_argument("--cache-embedder",
                        metavar="PATH",
                        type=Path,
                        default=None,
                        help="Cache fitted embedder path (query mode).")

    # ── Benchmark-only settings ───────────────────────────────────────────
    parser.add_argument("--n-examples",
                        type=int,
                        default=30,
                        metavar="N",
                        help="Examples per benchmark to download  [default: 30]",)
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Re-download even if JSON files already exist (benchmark mode).",)

    args = parser.parse_args()

    # Apply mode-specific defaults for thresholds / top-k
    if args.benchmarks is not None:
        args.sim_threshold   = args.sim_threshold   or 0.08
        args.score_threshold = args.score_threshold or 0.08
        args.top_k           = args.top_k           or 3
    else:
        args.sim_threshold   = args.sim_threshold   or 0.25
        args.score_threshold = args.score_threshold or 0.20
        args.top_k           = args.top_k           or 10

    return args
