# coding: utf-8
# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 12 examples: 4 easy · 4 medium · 4 hard
#
# Benchmark: AQuA-RAT (Algebra Question Answering with Rationales)
# Used in:   OPRO (arXiv:2309.03409)
#
# Design rationale — why this scenario produces a LOW baseline score
# ──────────────────────────────────────────────────────────────────
# Baseline skill: identifies the problem type and gives a rough estimate
# without performing algebraic steps or selecting the lettered option.
#
# The expected_behavior for every example includes a full step-by-step
# algebraic solution ending with "Answer: <letter>".  A response that
# says "this looks like a percentage problem, probably around 30" scores
# very low because it misses both the exact answer and the option letter.
#
# Easy baseline score:  ~0.10–0.20  (estimate near right ballpark, no letter)
# Hard baseline score:  ~0.02–0.10  (estimate wrong; letter absent)
#
# GEPA must learn to set up equations, show each algebraic step, evaluate
# the result, match it to the lettered options, and conclude with
# "Answer: <letter>".
# ══════════════════════════════════════════════════════════════════════════════

from .easy   import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard   import GOLDEN_EXAMPLES_HARD

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
