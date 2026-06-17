from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .recommender import build_recommender
from .runner_printer import _print_query_results, _print_skills


def _run_query(args: argparse.Namespace) -> None:
    """Production query mode — requires real GEPA scoring matrix files."""
    oracle_dir = (args.oracle_dir or args.data_dir).expanduser()
    cache_path = Path(args.cache_embedder).expanduser() if args.cache_embedder else None

    try:
        reccomender = build_recommender(oracle_dir=oracle_dir, variant=args.variant,
                                embedder_method=args.embedder)
    except FileNotFoundError as exc:
        sys.exit(
            f"[ERROR] {exc}\n\n"
            "Run gepa_scoring_matrix mode first with oracle_data_dir configured."
        )
    except ValueError as exc:
        sys.exit(
            f"[ERROR] {exc}\n\n"
            "The scoring_matrix_*.json files may predate baseline_cross_eval support.\n"
            "Re-run gepa_scoring_matrix to regenerate them."
        )

    if cache_path is not None:
        from .embedder import Embedder
        if cache_path.exists():
            try:
                reccomender._embedder = Embedder.load(cache_path)
                print(f"  Embedder cache   : loaded from {cache_path}")
            except Exception as exc:
                print(f"  [WARN] Could not load embedder cache ({exc}); using fresh model.")
        else:
            reccomender._embedder.save(cache_path)
            print(f"  Embedder cache   : saved to {cache_path}")

    print(f"  Loaded matrix: {reccomender.n_examples} rows · {len(reccomender.skills)} skill(s) · "
          f"{len(reccomender.metrics)} metric(s)")

    if args.list_skills:
        _print_skills(reccomender)
        return

    if args.from_file is not None:
        fp = Path(args.from_file).expanduser()
        if not fp.exists():
            sys.exit(f"[ERROR] File not found: {fp}")
        queries = [l.strip() for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not queries:
            sys.exit(f"[ERROR] File is empty: {fp}")
    elif args.query is None or args.query == "-":
        if sys.stdin.isatty():
            print("Enter your query (press Enter then Ctrl-D to submit):")
        raw = sys.stdin.read().strip()
        if not raw:
            sys.exit("Query is empty. Provide a prompt as argument, via --from-file, or stdin.")
        queries = [raw]
    else:
        queries = [args.query.strip()]

    for query in queries:
        if not query:
            continue
        results = reccomender.recommend(query=query,
                                        sim_threshold=args.sim_threshold,
                                        score_threshold=args.score_threshold,
                                        min_examples=args.min_examples,
                                        top_k=args.top_k)
        _print_query_results(results, query, args.variant)
