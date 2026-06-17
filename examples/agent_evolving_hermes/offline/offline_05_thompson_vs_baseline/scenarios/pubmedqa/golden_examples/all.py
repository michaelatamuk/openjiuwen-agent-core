# coding: utf-8
# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 12 examples: 4 easy · 4 medium · 4 hard
#
# Benchmark: PubMedQA
# Used in:   SkillGen (arXiv:2605.10999)
#
# Design rationale — why this scenario produces a LOW baseline score
# ──────────────────────────────────────────────────────────────────
# Baseline skill: outputs verbose medical disclaimers and hedged language
# instead of a clean verdict line followed by concise evidence.
#
# Expected_behavior for every example starts with a single verdict word
# (yes / no / maybe) on its own line, followed by a one-sentence evidence
# summary.  A response full of boilerplate ("always consult a professional…")
# scores very low because the signal-to-noise ratio is poor.
#
# Easy baseline score:  ~0.15–0.30  (verdict buried in hedging prose)
# Hard baseline score:  ~0.05–0.15  (verdict absent; wrong nuance)
#
# GEPA must learn to lead with the verdict, cite the key statistic, and
# omit all medical-disclaimer boilerplate.
# ══════════════════════════════════════════════════════════════════════════════

from .easy   import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard   import GOLDEN_EXAMPLES_HARD

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
