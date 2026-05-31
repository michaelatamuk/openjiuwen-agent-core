# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 20 hand-crafted examples: 4 easy · 8 medium · 8 hard
#
# Design rationale
# ────────────────
# The baseline api-security skill covers only surface-level checks
# (authentication presence, obvious injection, error leakage).
#
# Medium and hard examples (JWT specifics, SSRF, XXE, OAuth CSRF, timing
# attacks, ReDoS, mass assignment) are invisible to the shallow skill —
# the LLM-as-judge will score them near zero without the right prompt.
#
# TS (Level 2) learns which examples are most discriminating between
# evolved skill variants and focuses GEPA's budget on the hard ones.
# TS (Level 3) requires P(candidate > deployed) ≥ 0.75 before deploying.
#
# 20 examples → 50/25/25 split → train=10, val=5, holdout=5
# ══════════════════════════════════════════════════════════════════════════════
from .easy   import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard   import GOLDEN_EXAMPLES_HARD

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
