# coding: utf-8
"""
hotpotqa_loader.py — HotPotQA loader for skill_recommender

Loads the HotPotQA benchmark (``hotpotqa/hotpot_qa``, fullwiki setting)
and writes a ``scoring_matrix_hotpotqa.json`` file compatible with the
skill_recommender matrix format.

Used in:  DSPy (arXiv:2310.03714)

Dataset schema (distractor config — used here for fast download)
----------------------------------------------------------------
  question         : str
  answer           : str
  type             : str  ("bridge" | "comparison")
  level            : str  ("easy" | "medium" | "hard")
  context          : dict  (10 distractor + supporting paragraphs — not used)
  supporting_facts : dict  (sentence-level gold evidence — not used here)

Note: DSPy uses the ``fullwiki`` config which streams the full Wikipedia
index (~1 GB).  This loader uses ``distractor`` (same questions/answers,
pre-selected paragraphs, downloads in seconds).

Skill: ``multi-hop-qa``  (one skill; validation split used, test answers hidden)

Baseline strategy
-----------------
Return the last capitalised word(s) from the question — a crude
named-entity guess.  Bridge questions often ask about entities that
appear near the end ("What nationality is the director of X?"), so this
occasionally fires correctly and produces realistic F1/BoW variance.

Requirements
------------
    pip install datasets

Usage
-----
    from skill_recommender.hotpotqa_loader import load_hotpotqa_to_oracle
    load_hotpotqa_to_oracle(Path("/tmp/oracle"), n_examples=50)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .._scoring import FITNESS_METRICS, compute_scores

SKILL_NAME = "multi-hop-qa"


# ── Baseline ───────────────────────────────────────────────────────────────────

def _baseline_output(question: str) -> str:
    """Return the last sequence of capitalised / title-cased words in *question*."""
    # Find all capitalised tokens (skip first word which is always capitalised)
    tokens = question.split()
    caps   = [t.strip("?,.'\"") for t in tokens[1:] if t and t[0].isupper()]
    if caps:
        return caps[-1]
    # Fallback: last quoted phrase or last word
    quoted = re.findall(r'"([^"]+)"', question)
    if quoted:
        return quoted[-1]
    return tokens[-1].strip("?.,") if tokens else "unknown"


# ── Loader ─────────────────────────────────────────────────────────────────────

def _load_hotpotqa_rows(n_examples: int) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required.\n"
            "Install it with:  pip install datasets"
        ) from exc

    # streaming=True: fetch only n_examples without downloading the full split
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    rows = []
    for i, ex in enumerate(ds):
        if i >= n_examples:
            break
        rows.append({"question": ex["question"], "answer": ex["answer"]})
    return rows


def _build_payload(rows: list[dict]) -> dict:
    cross_eval = []
    for i, row in enumerate(rows):
        question = row["question"]
        expected = row["answer"]
        output   = _baseline_output(question)
        cross_eval.append({"example_id":       f"hotpotqa_{i:04d}",
                           "example_input":    question,
                           "example_expected": expected,
                           "candidate_output": output,
                           "scores":           compute_scores(output, expected),})
    return {"run_id":              "hotpotqa_validation",
            "skill_name":          SKILL_NAME,
            "fitness_metrics":     FITNESS_METRICS,
            "baseline_cross_eval": cross_eval,
            "evolved_cross_eval":  []}


# ── Public API ─────────────────────────────────────────────────────────────────

_DEFAULT_BUFFER = 500  # streaming buffer size for random sampling


def fetch_rows(n: int = 50, seed: int = 42, buffer_size: int = _DEFAULT_BUFFER) -> list[dict]:
    """Return *n* randomly-sampled HotPotQA validation rows (streaming).

    Each row is a dict with keys ``question``, ``answer``, and ``level``
    (``"easy"`` / ``"medium"`` / ``"hard"``).

    Uses streaming to avoid downloading the full ~500 MB split.  A buffer
    of ``buffer_size`` examples is fetched first, then ``n`` are sampled.

    This is the shared fetch primitive used by both the skill_recommender
    oracle builder and the demo scenario hf_loader.

    Parameters
    ----------
    n:
        Number of examples to return.
    seed:
        Random seed for reproducible sampling.
    buffer_size:
        How many streaming examples to buffer before sampling ``n``.
    """
    import itertools
    import random

    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required.\n"
            "Install it with:  pip install datasets"
        ) from exc

    ds = load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True
    )
    buffer = list(itertools.islice(ds, buffer_size))
    rng = random.Random(seed)
    sample = rng.sample(buffer, min(n, len(buffer)))
    return [
        {
            "question": r["question"],
            "answer": r["answer"],
            "level": r.get("level", "unknown"),
        }
        for r in sample
    ]


def load_hotpotqa_to_oracle(oracle_dir: Path, n_examples: int = 50, overwrite: bool = False) -> Path:
    """Download HotPotQA and write a scoring matrix JSON into *oracle_dir*.

    Parameters
    ----------
    oracle_dir : Path   Destination directory (created if absent).
    n_examples : int    Max examples from the validation split (default 50).
    overwrite  : bool   Replace existing file if True (default False).

    Returns
    -------
    Path  of the written JSON file.
    """
    oracle_dir = Path(oracle_dir).expanduser()
    oracle_dir.mkdir(parents=True, exist_ok=True)
    dest = oracle_dir / "scoring_matrix_hotpotqa.json"

    if dest.exists() and not overwrite:
        print("  skip  hotpotqa  (file exists, use overwrite=True to replace)")
        return dest

    print("  load  hotpotqa …", end=" ", flush=True)
    rows    = _load_hotpotqa_rows(n_examples)
    payload = _build_payload(rows)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"{len(rows)} examples → {dest.name}")
    return dest
