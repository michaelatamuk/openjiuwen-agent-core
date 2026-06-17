# coding: utf-8
"""
bbh_loader.py — BIG-Bench Hard loader for skill_recommender

Loads tasks from the BIG-Bench Hard benchmark (``lukaemon/bbh`` on HuggingFace)
and writes ``scoring_matrix_bbh_<task>.json`` files that are directly
compatible with the skill_recommender matrix format.

Each BBH task becomes one **skill**.  Because we do not run an LLM, we
compute scores against a simple deterministic baseline so the matrix
contains realistic (imperfect) numbers rather than placeholders.

Baseline strategy per task type
--------------------------------
  boolean     (web_of_lies, causal_judgement, …)   → always answer "True"
  navigate / sports_understanding                   → always answer "Yes"
  multiple-choice (A/B/C/D/E)                       → always pick "(A)"
  object_counting / penguins_in_a_table             → always answer "5"
  multistep_arithmetic_two                          → always answer "0"
  word_sorting                                      → sort words alphabetically
  date_understanding                                → "11/10/1969"
  default                                           → first word of question

Fitness metrics computed
------------------------
  exact_match   — 1.0 if normalised strings match, else 0.0
  f1            — token-level F1 (SQuAD-style)
  bag_of_words  — Jaccard similarity of word bags

Requirements
------------
    pip install datasets

Usage
-----
    from skill_recommender.bbh_loader import load_bbh_to_oracle

    load_bbh_to_oracle(
        oracle_dir  = Path("/tmp/my_oracle"),
        tasks       = ["logical_deduction_three_objects", "date_understanding"],
        n_examples  = 20,
    )
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Sequence

# ── Task catalogue ─────────────────────────────────────────────────────────────

# All 23 BBH task names in the lukaemon/bbh dataset
ALL_TASKS: list[str] = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

# Default subset used when no task list is given
DEFAULT_TASKS: list[str] = [
    "logical_deduction_three_objects",
    "date_understanding",
    "object_counting",
    "causal_judgement",
    "movie_recommendation",
    "word_sorting",
]

# ── Deterministic baseline ─────────────────────────────────────────────────────

_BOOLEAN_TASKS = frozenset(
    ["boolean_expressions", "web_of_lies", "formal_fallacies"]
)
_YES_TASKS = frozenset(
    ["causal_judgement", "navigate", "sports_understanding"]
)
_CHOICE_TASKS = frozenset(
    [
        "disambiguation_qa", "geometric_shapes", "hyperbaton",
        "logical_deduction_three_objects", "logical_deduction_five_objects",
        "logical_deduction_seven_objects", "movie_recommendation",
        "reasoning_about_colored_objects", "ruin_names",
        "salient_translation_error_detection", "temporal_sequences",
        "tracking_shuffled_objects_three_objects",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "dyck_languages",
    ]
)


def _baseline_output(task: str, inp: str) -> str:
    """Return a naive, non-LLM output for *inp* in *task*."""
    if task in _BOOLEAN_TASKS:
        return "True"
    if task in _YES_TASKS:
        return "Yes"
    if task in _CHOICE_TASKS:
        return "(A)"
    if task == "object_counting":
        return "5"
    if task == "penguins_in_a_table":
        return "5"
    if task == "multistep_arithmetic_two":
        return "0"
    if task == "date_understanding":
        return "11/10/1969"
    if task == "word_sorting":
        # Find the word list from the last line of the input
        last_line = inp.strip().split("\n")[-1]
        words = [w.strip(string.punctuation) for w in last_line.split()]
        return " ".join(sorted(set(w for w in words if w)))
    # Default: return first word of the question
    tokens = inp.split()
    return tokens[0] if tokens else ""


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def score_exact_match(output: str, expected: str) -> float:
    return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0


def score_f1(output: str, expected: str) -> float:
    pred   = _normalise(output)
    gold   = _normalise(expected)
    if not pred or not gold:
        return 0.0
    from collections import Counter
    common = Counter(pred) & Counter(gold)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred)
    recall    = n_common / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_bag_of_words(output: str, expected: str) -> float:
    pred = set(_normalise(output))
    gold = set(_normalise(expected))
    if not pred or not gold:
        return 0.0
    return len(pred & gold) / len(pred | gold)


FITNESS_METRICS = ["exact_match", "f1", "bag_of_words"]

_SCORERS = {
    "exact_match":  score_exact_match,
    "f1":           score_f1,
    "bag_of_words": score_bag_of_words,
}


def compute_scores(output: str, expected: str) -> dict[str, float]:
    return {m: fn(output, expected) for m, fn in _SCORERS.items()}


# ── Core loader ────────────────────────────────────────────────────────────────

def _load_bbh_task(task: str, n_examples: int) -> list[dict]:
    """Download one BBH task and return the first *n_examples* rows."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load BBH.\n"
            "Install it with:  pip install datasets"
        ) from exc

    ds = load_dataset("lukaemon/bbh", task, split="test")
    rows = []
    for i, example in enumerate(ds):
        if i >= n_examples:
            break
        rows.append({"input": example["input"], "target": example["target"]})
    return rows


def _build_matrix_payload(
    task: str,
    rows: list[dict],
) -> dict:
    """Convert raw BBH rows into a scoring_matrix_*.json payload."""
    cross_eval = []
    for i, row in enumerate(rows):
        inp      = row["input"]
        expected = row["target"]
        output   = _baseline_output(task, inp)
        scores   = compute_scores(output, expected)
        cross_eval.append(
            {
                "example_id":       f"{task}_{i:04d}",
                "example_input":    inp,
                "example_expected": expected,
                "candidate_output": output,
                "scores":           scores,
            }
        )

    return {
        "run_id":              f"bbh_{task}",
        "skill_name":          task.replace("_", "-"),
        "fitness_metrics":     FITNESS_METRICS,
        "baseline_cross_eval": cross_eval,
        "evolved_cross_eval":  [],      # no LLM run → empty
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def load_bbh_to_oracle(
    oracle_dir: Path,
    tasks: Sequence[str] | None = None,
    n_examples: int = 30,
    overwrite: bool = False,
) -> list[Path]:
    """Download BBH tasks and write scoring matrix JSON files into *oracle_dir*.

    Parameters
    ----------
    oracle_dir : Path
        Destination directory (created if absent).
    tasks : list[str] | None
        BBH task names to load.  Defaults to :data:`DEFAULT_TASKS`.
    n_examples : int
        Maximum examples to take from each task's test split (default 30).
    overwrite : bool
        If ``False`` (default), skip tasks whose file already exists.

    Returns
    -------
    list[Path]
        Paths of the written (or pre-existing) JSON files.
    """
    oracle_dir = Path(oracle_dir).expanduser()
    oracle_dir.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = DEFAULT_TASKS

    written: list[Path] = []
    for task in tasks:
        dest = oracle_dir / f"scoring_matrix_bbh_{task}.json"
        if dest.exists() and not overwrite:
            print(f"  skip  {task}  (file exists, use overwrite=True to replace)")
            written.append(dest)
            continue

        print(f"  load  {task} …", end=" ", flush=True)
        rows    = _load_bbh_task(task, n_examples)
        payload = _build_matrix_payload(task, rows)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"{len(rows)} examples → {dest.name}")
        written.append(dest)

    return written
