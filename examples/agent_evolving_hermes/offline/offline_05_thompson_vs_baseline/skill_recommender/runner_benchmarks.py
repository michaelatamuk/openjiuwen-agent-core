from __future__ import annotations

import argparse
import contextlib
import tempfile
from pathlib import Path

from .recommender import build_recommender
from .runner_args import DEFAULT_ORACLE_DIR
from .runner_printer import _print_benchmark_results


_BENCHMARK_QUERIES = [
    ("gsm8k",    "math-word-problems",
     "A baker made 48 cookies and packed them into boxes of 6. "
     "He sold 5 boxes. How many cookies does he have left?"),
    ("hotpotqa", "multi-hop-qa",
     "Who was the lead singer of the band that performed the theme song "
     "for the 1995 James Bond film?"),
    ("pubmedqa", "biomedical-qa",
     "Does regular physical exercise reduce the risk of type 2 diabetes "
     "in adults with pre-diabetic conditions?"),
    ("aquarat",  "algebra-reasoning",
     "If the price of a book is increased by 20% and then decreased by 10%, "
     "what is the net percentage change in the price? "
     "Options:\nA) 8% increase\nB) 10% decrease\nC) 8% decrease\nD) 10% increase\nE) No change"),
    ("bbh",      "<any bbh task>",
     "Sort the following words in alphabetical order: "
     "zebra mango apple pineapple cherry"),
]
ALL_BENCHMARKS      = ["bbh", "gsm8k", "hotpotqa", "pubmedqa", "aquarat"]


def _benchmark_loaders() -> dict:
    """Return a dict mapping benchmark name → oracle-loader callable."""
    from .data_loaders.bbh_loader import load_bbh_to_oracle, DEFAULT_TASKS as _BBH_DEFAULT
    from .data_loaders.gsm8k_loader import load_gsm8k_to_oracle        # noqa: E402
    from .data_loaders.hotpotqa_loader import load_hotpotqa_to_oracle  # noqa: E402
    from .data_loaders.pubmedqa_loader import load_pubmedqa_to_oracle  # noqa: E402
    from .data_loaders.aquarat_loader import load_aquarat_to_oracle    # noqa: E402

    return {
        "bbh":      lambda d, n, ow: load_bbh_to_oracle(d, tasks=_BBH_DEFAULT, n_examples=n, overwrite=ow),
        "gsm8k":    lambda d, n, ow: load_gsm8k_to_oracle(d, n_examples=n, overwrite=ow),
        "hotpotqa": lambda d, n, ow: load_hotpotqa_to_oracle(d, n_examples=n, overwrite=ow),
        "pubmedqa": lambda d, n, ow: load_pubmedqa_to_oracle(d, n_examples=n, overwrite=ow),
        "aquarat":  lambda d, n, ow: load_aquarat_to_oracle(d, n_examples=n, overwrite=ow),
    }


def _run_benchmarks(args: argparse.Namespace) -> None:
    """Download benchmarks, build recommender, show routing accuracy."""
    benchmarks = args.benchmarks if args.benchmarks else ALL_BENCHMARKS
    n_examples     = args.n_examples
    overwrite      = args.overwrite
    sim_threshold  = args.sim_threshold
    score_threshold= args.score_threshold
    top_k          = args.top_k

    oracle_dir_arg = (args.oracle_dir or args.data_dir).expanduser()

    ctx: contextlib.AbstractContextManager
    if oracle_dir_arg != DEFAULT_ORACLE_DIR:
        oracle_dir_arg.mkdir(parents=True, exist_ok=True)
        ctx = contextlib.nullcontext(oracle_dir_arg)
    else:
        ctx = tempfile.TemporaryDirectory(prefix="benchmarks_oracle_")  # type: ignore[assignment]

    with ctx as tmpdir:
        oracle_dir = Path(tmpdir)
        loaders = _benchmark_loaders()

        print(f"\nLoading benchmarks → {oracle_dir}")
        for name in benchmarks:
            if name not in loaders:
                print(f"  [WARN] Unknown benchmark '{name}', skipping.")
                continue
            loaders[name](oracle_dir, n_examples, overwrite)

        print("\nBuilding recommender …")
        rec = build_recommender(oracle_dir=oracle_dir, variant="baseline",
                                embedder_method="tfidf")
        print(f"  Rows    : {rec.n_examples}")
        print(f"  Skills  : {rec.skills}")
        print(f"  Metrics : {rec.metrics}")

        loaded_skills = set(rec.skills)
        correct = total = 0

        for bench, skill_name, query in _BENCHMARK_QUERIES:
            if bench not in benchmarks:
                continue
            results = rec.recommend(query=query, sim_threshold=sim_threshold,
                                    score_threshold=score_threshold, top_k=top_k)
            if bench == "bbh":
                is_hit = bool(results and results[0]["skill"] in loaded_skills)
                _print_benchmark_results(results, query, "<any bbh task>",
                                         is_hit=is_hit)
            else:
                if skill_name not in loaded_skills:
                    continue
                _print_benchmark_results(results, query, skill_name)
                is_hit = bool(results and results[0]["skill"] == skill_name)
            if is_hit:
                correct += 1
            total += 1

        if total:
            print(f"\n  Routing accuracy: {correct}/{total} queries routed to expected skill")
