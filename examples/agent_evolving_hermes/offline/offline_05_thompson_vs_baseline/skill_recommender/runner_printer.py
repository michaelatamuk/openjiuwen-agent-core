from __future__ import annotations

from .recommender import SkillRecommender


SEP  = "─" * 72
SEP2 = "═" * 72


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
