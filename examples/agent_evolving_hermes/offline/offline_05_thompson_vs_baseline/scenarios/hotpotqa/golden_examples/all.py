# coding: utf-8
# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 12 examples: 4 easy · 4 medium · 4 hard
#
# Benchmark: HotPotQA (multi-hop question answering)
# Used in:   DSPy (arXiv:2310.03714)
#
# Design rationale — why this scenario produces a LOW baseline score
# ──────────────────────────────────────────────────────────────────
# Baseline skill: outputs a bare final answer without showing any
# intermediate reasoning hops.
#
# The expected_behavior for every example identifies each bridge entity
# explicitly, names the supporting fact, and only then states the answer.
# A baseline output of "Paris" scores ~0.20 against an expected output
# that includes "The Eiffel Tower is located in France. The capital of
# France is Paris."
#
# Hard (comparison) examples are worst for the baseline — it picks one
# entity without surfacing the attribute values being compared.
#
# Easy baseline score  : ~0.20–0.35  (answer word present, chain absent)
# Hard baseline score  : ~0.05–0.15  (comparison logic entirely missing)
# ══════════════════════════════════════════════════════════════════════════════

from .easy   import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard   import GOLDEN_EXAMPLES_HARD

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
