# coding: utf-8
# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 12 examples: 4 easy · 4 medium · 4 hard
#
# Benchmark: GSM8K (Grade School Math 8K)
# Used in:   OPRO (arXiv:2309.03409), DSPy (arXiv:2310.03714)
#
# Design rationale — why this scenario produces a LOW baseline score
# ──────────────────────────────────────────────────────────────────
# Baseline skill: outputs only a brief final answer with no working.
#
# The expected_behavior for every example includes a full step-by-step
# reasoning chain ending with a bolded answer.  A skill that only outputs
# "15 cookies" scores very low (short answer vs long working chain).
#
# Easy baseline score:  ~0.20–0.35  (answer word present but no chain)
# Hard baseline score:  ~0.05–0.15  (wrong or missing intermediate steps)
#
# GEPA must learn to show every arithmetic step, label intermediate
# results, and format the final answer clearly.
# ══════════════════════════════════════════════════════════════════════════════

from .easy   import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard   import GOLDEN_EXAMPLES_HARD

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
