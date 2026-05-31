# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 20 hand-crafted examples: 4 easy · 8 medium · 8 hard
#
# Design rationale: easy examples are visible even to a shallow skill.
# Medium and hard examples (security bugs, concurrency, N+1) only surface in
# a genuinely deep review.  TS will learn to focus budget on these hard
# examples, which are most discriminating between evolved variants.
#
# 20 examples → 50/25/25 split → train=10, val=5, holdout=5
# 5 holdout examples give the TS Level 3 acceptance gate enough statistical
# power to reach 75% confidence when there is a genuine improvement.
# ══════════════════════════════════════════════════════════════════════════════
from .easy import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard import GOLDEN_EXAMPLES_HARD


GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
