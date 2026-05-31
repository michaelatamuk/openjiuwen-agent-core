# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 20 hand-crafted examples: 4 easy · 8 medium · 8 hard
#
# Design rationale — why this scenario produces a LOW baseline score
# ──────────────────────────────────────────────────────────────────
# The baseline skill says only "check for bugs and logical errors" with no
# mention of machine learning, data leakage, or sklearn APIs.
#
# Hard data-leakage bugs LOOK correct to a generic reviewer:
#   - fit_transform(X) before split:  valid Python, no exception
#   - KFold on grouped patient data:  "best practice" cross-validation
#   - SMOTE before split:             "the class balance is fixed"
#   - shift(-1) as feature:           "uses recent data"
#
# Baseline LLM output on hard examples:  "looks good, add comments"
# LLM-as-judge score for that output:    0.05–0.15
# Expected baseline holdout score:       ~0.25–0.35
#
# After evolution the skill guides toward data_leakage checks, Pipeline
# usage, GroupKFold, TimeSeriesSplit, and SMOTE-inside-fold patterns.
# Expected evolved holdout score:        ~0.65–0.80
# ══════════════════════════════════════════════════════════════════════════════
from .easy   import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard   import GOLDEN_EXAMPLES_HARD

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
