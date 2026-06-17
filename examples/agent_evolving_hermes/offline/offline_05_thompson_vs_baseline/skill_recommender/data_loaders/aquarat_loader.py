# coding: utf-8
"""
aquarat_loader.py — AQuA-RAT loader for skill_recommender

Loads the AQuA-RAT benchmark (``deepmind/aqua_rat``, raw config)
and writes a ``scoring_matrix_aquarat.json`` file compatible with the
skill_recommender matrix format.

Used in:  OPRO (arXiv:2309.03409) as a GSM8K transfer benchmark

Dataset schema (raw / test split)
----------------------------------
  question          : str   — algebra word problem
  options           : list  — ["A)12", "B)24", "C)36", "D)48", "E)60"]
  rationale         : str   — step-by-step solution
  correct           : str   — "B" (the correct option letter)
  annotated_formula : str

Skill: ``algebra-reasoning``

Input format
------------
The question and all options are joined so the recommender can distinguish
algebra prompts from plain arithmetic (GSM8K):

    "<question>\\nOptions:\\nA) ...\\nB) ...\\n..."

Baseline strategy
-----------------
Always answer "A".  With 5 choices the expected accuracy is ~20%, producing
realistic low-but-nonzero scores that contrast well with PubMedQA's ~55%.

Requirements
------------
    pip install datasets

Usage
-----
    from skill_recommender.aquarat_loader import load_aquarat_to_oracle
    load_aquarat_to_oracle(Path("/tmp/oracle"), n_examples=50)
"""
from __future__ import annotations

import json
from pathlib import Path

from .._scoring import FITNESS_METRICS, compute_scores

SKILL_NAME = "algebra-reasoning"
_BASELINE  = "A"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_input(question: str, options: list) -> str:
    opts = "\n".join(str(o) for o in options)
    return f"{question}\nOptions:\n{opts}"


# ── Loader ─────────────────────────────────────────────────────────────────────

def _load_aquarat_rows(n_examples: int) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required.\n"
            "Install it with:  pip install datasets"
        ) from exc

    ds = load_dataset("deepmind/aqua_rat", "raw", split="test")
    rows = []
    for i, ex in enumerate(ds):
        if i >= n_examples:
            break
        rows.append({
            "question": ex["question"],
            "options":  ex["options"],
            "correct":  ex["correct"],
        })
    return rows


def _build_payload(rows: list[dict]) -> dict:
    cross_eval = []
    for i, row in enumerate(rows):
        inp      = _format_input(row["question"], row["options"])
        expected = row["correct"].strip().upper()
        output   = _BASELINE
        cross_eval.append({
            "example_id":       f"aquarat_{i:04d}",
            "example_input":    inp,
            "example_expected": expected,
            "candidate_output": output,
            "scores":           compute_scores(output, expected),
        })
    return {
        "run_id":              "aquarat_test",
        "skill_name":          SKILL_NAME,
        "fitness_metrics":     FITNESS_METRICS,
        "baseline_cross_eval": cross_eval,
        "evolved_cross_eval":  [],
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_rows(n: int = 50, seed: int = 42) -> list[dict]:
    """Return *n* randomly-sampled AQuA-RAT test rows.

    Each row is a dict with keys ``question``, ``options`` (list of strings
    like ``"A)120"``), ``rationale`` (step-by-step solution), and ``correct``
    (option letter string, e.g. ``"B"``).

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

    ds = load_dataset("deepmind/aqua_rat", "raw", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [
        {
            "question": r["question"],
            "options": r["options"],
            "rationale": r.get("rationale", ""),
            "correct": r["correct"],
        }
        for r in ds
    ]


def load_aquarat_to_oracle(
    oracle_dir: Path,
    n_examples: int = 50,
    overwrite: bool = False,
) -> Path:
    """Download AQuA-RAT and write a scoring matrix JSON into *oracle_dir*.

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
    dest = oracle_dir / "scoring_matrix_aquarat.json"

    if dest.exists() and not overwrite:
        print("  skip  aquarat  (file exists, use overwrite=True to replace)")
        return dest

    print("  load  aquarat …", end=" ", flush=True)
    rows    = _load_aquarat_rows(n_examples)
    payload = _build_payload(rows)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"{len(rows)} examples → {dest.name}")
    return dest
