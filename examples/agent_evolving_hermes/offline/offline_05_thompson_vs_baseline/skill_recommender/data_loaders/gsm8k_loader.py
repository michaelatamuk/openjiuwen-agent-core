# coding: utf-8
"""
gsm8k_loader.py — GSM8K loader for skill_recommender

Loads the Grade School Math 8K benchmark (``openai/gsm8k`` on HuggingFace)
and writes a ``scoring_matrix_gsm8k.json`` file compatible with the
skill_recommender matrix format.

Used in:  OPRO (arXiv:2309.03409), DSPy (arXiv:2310.03714)

Dataset schema
--------------
  question : str   — the math word problem
  answer   : str   — chain-of-thought ending with "#### <number>"

Skill: ``math-word-problems``  (one skill, 1 319 test examples)

Baseline strategy
-----------------
Extract every integer that appears in the question text and return their
sum.  This is almost always wrong (the correct answer usually differs)
but produces realistic, varied scores per example — some accidental
exact-match hits, moderate BoW/F1 when the answer is a number that also
appeared in the question.

Requirements
------------
    pip install datasets

Usage
-----
    from skill_recommender.gsm8k_loader import load_gsm8k_to_oracle
    load_gsm8k_to_oracle(Path("/tmp/oracle"), n_examples=50)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from .._scoring import FITNESS_METRICS, compute_scores

SKILL_NAME = "math-word-problems"


# ── Baseline ───────────────────────────────────────────────────────────────────

def _extract_answer(answer_text: str) -> str:
    """Pull the final number after '####' in a GSM8K answer string."""
    m = re.search(r"####\s*([^\n]+)", answer_text)
    return m.group(1).strip() if m else answer_text.strip()


def _baseline_output(question: str) -> str:
    """Return the sum of all integers found in the question as a string."""
    numbers = re.findall(r"\b\d+\b", question)
    if not numbers:
        return "0"
    return str(sum(int(n) for n in numbers))


# ── Loader ─────────────────────────────────────────────────────────────────────

def _load_gsm8k_rows(n_examples: int) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required.\n"
            "Install it with:  pip install datasets"
        ) from exc

    ds = load_dataset("openai/gsm8k", "main", split="test")
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
        expected = _extract_answer(row["answer"])
        output   = _baseline_output(question)
        cross_eval.append({
            "example_id":       f"gsm8k_{i:04d}",
            "example_input":    question,
            "example_expected": expected,
            "candidate_output": output,
            "scores":           compute_scores(output, expected),
        })
    return {
        "run_id":              "gsm8k_test",
        "skill_name":          SKILL_NAME,
        "fitness_metrics":     FITNESS_METRICS,
        "baseline_cross_eval": cross_eval,
        "evolved_cross_eval":  [],
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_rows(n: int = 50, seed: int = 42) -> list[dict]:
    """Return *n* randomly-sampled GSM8K test rows.

    Each row is a dict with keys ``question`` and ``answer`` (the full
    chain-of-thought string ending with ``#### <number>``).

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

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [{"question": r["question"], "answer": r["answer"]} for r in ds]


def load_gsm8k_to_oracle(
    oracle_dir: Path,
    n_examples: int = 50,
    overwrite: bool = False,
) -> Path:
    """Download GSM8K and write a scoring matrix JSON into *oracle_dir*.

    Parameters
    ----------
    oracle_dir : Path   Destination directory (created if absent).
    n_examples : int    Max examples from the test split (default 50).
    overwrite  : bool   Replace existing file if True (default False).

    Returns
    -------
    Path  of the written JSON file.
    """
    oracle_dir = Path(oracle_dir).expanduser()
    oracle_dir.mkdir(parents=True, exist_ok=True)
    dest = oracle_dir / "scoring_matrix_gsm8k.json"

    if dest.exists() and not overwrite:
        print(f"  skip  gsm8k  (file exists, use overwrite=True to replace)")
        return dest

    print("  load  gsm8k …", end=" ", flush=True)
    rows    = _load_gsm8k_rows(n_examples)
    payload = _build_payload(rows)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"{len(rows)} examples → {dest.name}")
    return dest
