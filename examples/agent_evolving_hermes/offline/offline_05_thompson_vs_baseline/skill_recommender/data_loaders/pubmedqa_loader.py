# coding: utf-8
"""
pubmedqa_loader.py — PubMedQA loader for skill_recommender

Loads the PubMedQA benchmark (``qiaojin/PubMedQA``, pqa_labeled config)
and writes a ``scoring_matrix_pubmedqa.json`` file compatible with the
skill_recommender matrix format.

Used in:  SkillGen (arXiv:2605.10999)

Dataset schema (pqa_labeled / train split — 1 000 expert-labelled examples)
-----------------------------------------------------------------------------
  pubid          : int
  question       : str
  context        : dict  (supporting abstract sentences — not used for input)
  long_answer    : str
  final_decision : str   "yes" | "no" | "maybe"

Skill: ``biomedical-qa``

Baseline strategy
-----------------
Always answer "yes".  The PubMedQA label distribution is roughly
55% yes / 34% no / 11% maybe, so this baseline reaches ~55% exact match
and realistic F1/BoW scores — a reasonable uninformed prior.

Requirements
------------
    pip install datasets

Usage
-----
    from skill_recommender.pubmedqa_loader import load_pubmedqa_to_oracle
    load_pubmedqa_to_oracle(Path("/tmp/oracle"), n_examples=50)
"""
from __future__ import annotations

import json
from pathlib import Path

from .._scoring import FITNESS_METRICS, compute_scores

SKILL_NAME = "biomedical-qa"
_BASELINE  = "yes"


# ── Loader ─────────────────────────────────────────────────────────────────────

def _load_pubmedqa_rows(n_examples: int) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required.\n"
            "Install it with:  pip install datasets"
        ) from exc

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = []
    for i, ex in enumerate(ds):
        if i >= n_examples:
            break
        rows.append({
            "question":       ex["question"],
            "final_decision": ex["final_decision"],
        })
    return rows


def _build_payload(rows: list[dict]) -> dict:
    cross_eval = []
    for i, row in enumerate(rows):
        question = row["question"]
        expected = row["final_decision"].strip().lower()
        output   = _BASELINE
        cross_eval.append({
            "example_id":       f"pubmedqa_{i:04d}",
            "example_input":    question,
            "example_expected": expected,
            "candidate_output": output,
            "scores":           compute_scores(output, expected),
        })
    return {
        "run_id":              "pubmedqa_labeled",
        "skill_name":          SKILL_NAME,
        "fitness_metrics":     FITNESS_METRICS,
        "baseline_cross_eval": cross_eval,
        "evolved_cross_eval":  [],
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_rows(n: int = 50, seed: int = 42) -> list[dict]:
    """Return *n* randomly-sampled PubMedQA pqa_labeled rows.

    Each row is a dict with keys ``question``, ``context`` (the original
    HuggingFace context dict with ``contexts`` list of abstract sentences),
    ``final_decision`` (``"yes"`` / ``"no"`` / ``"maybe"``), and
    ``long_answer`` (expert summary string).

    This is the shared fetch primitive used by both the skill_recommender
    oracle builder and the demo scenario hf_loader.

    Parameters
    ----------
    n:
        Number of examples to return.
    seed:
        Random seed for reproducible sampling.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required.\n"
            "Install it with:  pip install datasets"
        ) from exc

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [
        {
            "question": r["question"],
            "context": r["context"],
            "final_decision": r["final_decision"],
            "long_answer": r.get("long_answer", ""),
        }
        for r in ds
    ]


def load_pubmedqa_to_oracle(
    oracle_dir: Path,
    n_examples: int = 50,
    overwrite: bool = False,
) -> Path:
    """Download PubMedQA and write a scoring matrix JSON into *oracle_dir*.

    Parameters
    ----------
    oracle_dir : Path   Destination directory (created if absent).
    n_examples : int    Max examples from the labelled train split (default 50).
    overwrite  : bool   Replace existing file if True (default False).

    Returns
    -------
    Path  of the written JSON file.
    """
    oracle_dir = Path(oracle_dir).expanduser()
    oracle_dir.mkdir(parents=True, exist_ok=True)
    dest = oracle_dir / "scoring_matrix_pubmedqa.json"

    if dest.exists() and not overwrite:
        print("  skip  pubmedqa  (file exists, use overwrite=True to replace)")
        return dest

    print("  load  pubmedqa …", end=" ", flush=True)
    rows    = _load_pubmedqa_rows(n_examples)
    payload = _build_payload(rows)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"{len(rows)} examples → {dest.name}")
    return dest
