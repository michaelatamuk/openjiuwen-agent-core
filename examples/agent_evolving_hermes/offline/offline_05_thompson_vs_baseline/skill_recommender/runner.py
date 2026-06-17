#!/usr/bin/env python3
# coding: utf-8
"""
runner.py — Skill Recommender  (single entry point)

Three modes selected by flags:

  QUERY MODE  (default — no mode flag)
  ─────────────────────────────────────
  Given a prompt, finds similar examples in the scoring matrix and recommends
  skills and their best-performing fitness metrics.

      python runner.py "How do I reset my SmartHub?"
      echo "Explain invoice terms" | python runner.py -
      python runner.py --from-file queries.txt
      python runner.py --list-skills
      python runner.py "some prompt" --cache-embedder /tmp/emb.pkl

  DEMO MODE  (--demo)
  ────────────────────
  Zero-network quick start.  Creates a small in-memory synthetic scoring
  matrix and runs three sample queries so you can see the output format
  without a real GEPA run.

      python runner.py --demo

  BENCHMARK MODE  (--benchmarks [NAME ...])
  ──────────────────────────────────────────
  Downloads benchmark datasets from HuggingFace, builds a scoring matrix
  from deterministic baselines, and shows routing accuracy.

      python runner.py --benchmarks                     # all 5 benchmarks
      python runner.py --benchmarks gsm8k hotpotqa      # specific subset
      python runner.py --benchmarks bbh --oracle-dir /tmp/bbh_oracle

  Available benchmarks: bbh · gsm8k · hotpotqa · pubmedqa · aquarat

Common options
──────────────
  --data-dir PATH        Scoring-matrix directory  [default: ~/.openjiuwen/oracle]
  --oracle-dir PATH      Alias for --data-dir (accepted in all modes)
  --variant              baseline | evolved | both  [default: baseline]
  --embedder             tfidf | openai  [default: tfidf]
  --sim-threshold FLOAT  [default: 0.25 / 0.08 in benchmark mode]
  --score-threshold FLOAT[default: 0.20 / 0.08 in benchmark mode]
  --top-k N              [default: 10 / 3 in benchmark mode]

Benchmark-only options
──────────────────────
  --n-examples N         Examples per benchmark to download  [default: 30]
  --overwrite            Re-download even if JSON files already exist

Query-only options
──────────────────
  --cache-embedder PATH  Cache fitted TF-IDF embedder (load if exists, save otherwise)
  --min-examples N       Min similar examples required  [default: 1]

Run as module
─────────────
  python -m skill_recommender [ARGS]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path

# ── path bootstrap (allows running as a plain script) ─────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent.parent.parent  # michael-agent-core/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.recommender import (  # noqa: E402
    build_recommender,
    SkillRecommender,
)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_ORACLE_DIR  = Path("~/.openjiuwen/oracle").expanduser()
ALL_BENCHMARKS      = ["bbh", "gsm8k", "hotpotqa", "pubmedqa", "aquarat"]

SEP  = "─" * 72
SEP2 = "═" * 72


# ── Shared output helpers ─────────────────────────────────────────────────────

def _print_query_results(results: list[dict], query: str, variant: str) -> None:
    """Detailed output for query mode — shows similar examples."""
    print(f"\n{SEP2}")
    print(f"  Skill Recommender  [{variant}]")
    print(f"  Query  : {query[:120]}")
    print(SEP2)
    if not results:
        print("  No recommendations above the configured thresholds.")
        print(SEP2)
        return
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] Skill  : {r['skill']}")
        print(f"       Metric : {r['metric']}")
        print(f"       Score  : {r['score']:.3f}  "
              f"(mean_sim={r['mean_similarity']:.3f}, "
              f"{r['n_examples']} similar example(s))")
        for ex in r["similar_examples"]:
            inp = ex.get("input", "")
            exp = ex.get("expected", "")
            out = ex.get("output", "")
            print(f"       ↳ [{ex['similarity']:.3f}] prompt   : {inp}")
            if exp:
                print(f"                 expected : {exp}")
            if out:
                print(f"                 output   : {out}")
    print(f"\n{SEP}")


def _print_benchmark_results(
    results: list[dict],
    query: str,
    expected_skill: str,
    is_hit: bool | None = None,
) -> None:
    """Compact output for benchmark mode — shows top-1 routing hit/miss."""
    short = query if len(query) <= 78 else query[:75] + "…"
    print(f"\n{SEP2}")
    print(f"  Query    : {short}")
    print(f"  Expected : {expected_skill}")
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
    print(f"\n  Top-1: {hit}  ({top['skill']})")
    print(f"\n{SEP}")


def _print_skills(rec: SkillRecommender) -> None:
    print(f"\n{SEP2}")
    print("  Skills in the matrix")
    print(SEP2)
    print(f"  Total rows : {rec.n_examples}")
    print(f"  Metrics    : {rec.metrics}")
    print()
    for skill in rec.skills:
        n = int((rec._df["skill_name"] == skill).sum())
        print(f"  {skill:<40s}  {n:>4} example(s)")
    print(SEP)


# ── Demo mode ─────────────────────────────────────────────────────────────────

_DEMO_MATRIX = [
    {
        "run_id":          "demo_smarthub_001",
        "skill_name":      "smarthub-support",
        "fitness_metrics": ["bag_of_words", "f1", "format"],
        "baseline_cross_eval": [
            {
                "example_id":       "sh_01",
                "example_input":    "How do I reset my SmartHub device to factory settings?",
                "example_expected": "Hold the reset button for 10 seconds until the LED flashes red.",
                "candidate_output": "Press and hold the reset button on the back for 10 seconds.",
                "scores": {"bag_of_words": 0.82, "f1": 0.74, "format": 0.90},
            },
            {
                "example_id":       "sh_02",
                "example_input":    "My SmartHub keeps disconnecting from WiFi, what should I do?",
                "example_expected": "Restart the hub and check that the firmware is up to date.",
                "candidate_output": "Try restarting the hub. Also make sure the firmware is current.",
                "scores": {"bag_of_words": 0.78, "f1": 0.71, "format": 0.85},
            },
            {
                "example_id":       "sh_03",
                "example_input":    "How do I add a new device to my SmartHub network?",
                "example_expected": "Open the SmartHub app and tap 'Add Device', then follow the prompts.",
                "candidate_output": "In the SmartHub app, go to Add Device and follow the on-screen steps.",
                "scores": {"bag_of_words": 0.88, "f1": 0.81, "format": 0.92},
            },
            {
                "example_id":       "sh_04",
                "example_input":    "What does the blinking orange light on my hub mean?",
                "example_expected": "Blinking orange means the hub is searching for an internet connection.",
                "candidate_output": "A blinking orange LED indicates the hub is trying to connect to the internet.",
                "scores": {"bag_of_words": 0.80, "f1": 0.76, "format": 0.88},
            },
            {
                "example_id":       "sh_05",
                "example_input":    "Can I connect my SmartHub to a 5GHz WiFi network?",
                "example_expected": "Yes, the SmartHub supports both 2.4GHz and 5GHz bands.",
                "candidate_output": "The SmartHub supports both 2.4 GHz and 5 GHz wireless networks.",
                "scores": {"bag_of_words": 0.75, "f1": 0.69, "format": 0.80},
            },
        ],
    },
    {
        "run_id":          "demo_billing_001",
        "skill_name":      "billing-support",
        "fitness_metrics": ["bag_of_words", "f1", "format"],
        "baseline_cross_eval": [
            {
                "example_id":       "bi_01",
                "example_input":    "How do I update my payment method?",
                "example_expected": "Log in, go to Account → Billing, then click Change Payment Method.",
                "candidate_output": "Sign in and navigate to Account > Billing > Change Payment Method.",
                "scores": {"bag_of_words": 0.84, "f1": 0.79, "format": 0.91},
            },
            {
                "example_id":       "bi_02",
                "example_input":    "Why was I charged twice this month?",
                "example_expected": "This can happen if two subscriptions overlap. Contact support to resolve.",
                "candidate_output": "Double charges usually mean overlapping subscriptions. Please contact us.",
                "scores": {"bag_of_words": 0.70, "f1": 0.65, "format": 0.78},
            },
            {
                "example_id":       "bi_03",
                "example_input":    "How do I download my invoice as a PDF?",
                "example_expected": "Go to Account → Billing History and click the PDF icon next to any invoice.",
                "candidate_output": "In Account > Billing History you can click the PDF icon to download invoices.",
                "scores": {"bag_of_words": 0.86, "f1": 0.80, "format": 0.93},
            },
            {
                "example_id":       "bi_04",
                "example_input":    "What payment methods do you accept?",
                "example_expected": "We accept Visa, Mastercard, Amex, PayPal, and bank transfer.",
                "candidate_output": "Accepted payment methods: Visa, Mastercard, Amex, PayPal, and bank transfer.",
                "scores": {"bag_of_words": 0.91, "f1": 0.87, "format": 0.95},
            },
            {
                "example_id":       "bi_05",
                "example_input":    "Can I get a refund for my last subscription payment?",
                "example_expected": "Refunds are available within 14 days of payment. Submit a refund request.",
                "candidate_output": "If the payment was made within 14 days, you can request a refund.",
                "scores": {"bag_of_words": 0.77, "f1": 0.72, "format": 0.82},
            },
        ],
    },
]

_DEMO_QUERIES = [
    "My router keeps dropping the connection, how do I fix it?",
    "I need a PDF copy of my latest invoice",
    "What lights should I see when the hub is working normally?",
]


def _run_demo() -> None:
    """Synthetic quick-start demo — no network or GEPA run required."""
    with tempfile.TemporaryDirectory(prefix="skill_recommender_demo_") as tmpdir:
        oracle_dir = Path(tmpdir)

        for entry in _DEMO_MATRIX:
            fname = f"scoring_matrix_{entry['run_id']}.json"
            with open(oracle_dir / fname, "w", encoding="utf-8") as fh:
                json.dump(entry, fh, indent=2, ensure_ascii=False)

        print(f"\n  Demo matrix written to: {oracle_dir}")
        rec = build_recommender(oracle_dir=oracle_dir, variant="baseline",
                                embedder_method="tfidf")
        print(f"  Loaded  : {rec.n_examples} rows · "
              f"{len(rec.skills)} skill(s) · {len(rec.metrics)} metric(s)")
        print(f"  Skills  : {rec.skills}")
        print(f"  Metrics : {rec.metrics}")

        for query in _DEMO_QUERIES:
            results = rec.recommend(query=query, sim_threshold=0.10,
                                    score_threshold=0.15, top_k=3)
            _print_query_results(results, query, "baseline")


# ── Benchmark mode ────────────────────────────────────────────────────────────

# Sample queries for benchmark routing accuracy check — one per benchmark domain.
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


def _benchmark_loaders() -> dict:
    """Return a dict mapping benchmark name → oracle-loader callable."""
    from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.bbh_loader import (  # noqa: E402
        load_bbh_to_oracle, DEFAULT_TASKS as _BBH_DEFAULT,
    )
    from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.gsm8k_loader import load_gsm8k_to_oracle        # noqa: E402
    from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.hotpotqa_loader import load_hotpotqa_to_oracle  # noqa: E402
    from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.pubmedqa_loader import load_pubmedqa_to_oracle  # noqa: E402
    from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.data_loaders.aquarat_loader import load_aquarat_to_oracle    # noqa: E402

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


# ── Query mode ────────────────────────────────────────────────────────────────

def _run_query(args: argparse.Namespace) -> None:
    """Production query mode — requires real GEPA scoring matrix files."""
    oracle_dir = (args.oracle_dir or args.data_dir).expanduser()
    cache_path = Path(args.cache_embedder).expanduser() if args.cache_embedder else None

    try:
        rec = build_recommender(oracle_dir=oracle_dir, variant=args.variant,
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
        from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.skill_recommender.embedder import Embedder  # noqa: E402
        if cache_path.exists():
            try:
                rec._embedder = Embedder.load(cache_path)
                print(f"  Embedder cache   : loaded from {cache_path}")
            except Exception as exc:
                print(f"  [WARN] Could not load embedder cache ({exc}); using fresh model.")
        else:
            rec._embedder.save(cache_path)
            print(f"  Embedder cache   : saved to {cache_path}")

    print(f"  Loaded matrix: {rec.n_examples} rows · {len(rec.skills)} skill(s) · "
          f"{len(rec.metrics)} metric(s)")

    if args.list_skills:
        _print_skills(rec)
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
        results = rec.recommend(query=query, sim_threshold=args.sim_threshold,
                                score_threshold=args.score_threshold,
                                min_examples=args.min_examples, top_k=args.top_k)
        _print_query_results(results, query, args.variant)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="runner.py",
        description="Skill Recommender — query, demo, or benchmark mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode flags ────────────────────────────────────────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run the synthetic quick-start demo (no network required).",
    )
    mode_group.add_argument(
        "--benchmarks",
        nargs="*",
        metavar="NAME",
        default=None,
        help=f"Download HuggingFace benchmarks and show routing accuracy. "
             f"Pass names to select a subset, or omit names for all. "
             f"Available: {', '.join(ALL_BENCHMARKS)}",
    )

    # ── Query input (query mode only) ─────────────────────────────────────
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Prompt to find skills for (query mode).  Pass '-' to read from stdin.",
    )
    query_group.add_argument(
        "--from-file",
        metavar="PATH",
        type=Path,
        default=None,
        help="File with one query per line (query mode).",
    )
    query_group.add_argument(
        "--list-skills",
        action="store_true",
        default=False,
        help="List all skills in the matrix and exit (query mode).",
    )

    # ── Shared settings ───────────────────────────────────────────────────
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        type=Path,
        default=DEFAULT_ORACLE_DIR,
        help="Directory with scoring_matrix_*.json files  [default: ~/.openjiuwen/oracle]",
    )
    parser.add_argument(
        "--oracle-dir",
        metavar="PATH",
        type=Path,
        default=None,
        help="Alias for --data-dir (accepted in all modes).",
    )
    parser.add_argument(
        "--variant",
        choices=["baseline", "evolved", "both"],
        default="baseline",
        help="Matrix layer to use  [default: baseline]",
    )
    parser.add_argument(
        "--embedder",
        choices=["tfidf", "openai"],
        default="tfidf",
        help="Embedding backend  [default: tfidf]",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Min cosine similarity  [default: 0.25 for query, 0.08 for benchmarks]",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Min weighted score  [default: 0.20 for query, 0.08 for benchmarks]",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="N",
        help="Max results per query  [default: 10 for query, 3 for benchmarks]",
    )

    # ── Query-only settings ───────────────────────────────────────────────
    parser.add_argument(
        "--min-examples",
        type=int,
        default=1,
        metavar="N",
        help="Min similar examples required (query mode)  [default: 1]",
    )
    parser.add_argument(
        "--cache-embedder",
        metavar="PATH",
        type=Path,
        default=None,
        help="Cache fitted embedder path (query mode).",
    )

    # ── Benchmark-only settings ───────────────────────────────────────────
    parser.add_argument(
        "--n-examples",
        type=int,
        default=30,
        metavar="N",
        help="Examples per benchmark to download  [default: 30]",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if JSON files already exist (benchmark mode).",
    )

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

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.demo:
        _run_demo()
    elif args.benchmarks is not None:
        _run_benchmarks(args)
    else:
        _run_query(args)


if __name__ == "__main__":
    main()
