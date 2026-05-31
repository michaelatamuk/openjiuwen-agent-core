# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 20 hand-crafted examples: 4 easy · 8 medium · 8 hard
#
# Design rationale — why this scenario produces a VERY LOW baseline score
# ────────────────────────────────────────────────────────────────────────
# Baseline skill: copy-editor / formatting checker only.
#   Explicitly instructed NOT to evaluate statistical methodology or research
#   design validity.
#
# Hard methodology issues are INVISIBLE to a formatting-only reviewer:
#   - HARKing:                 "introduction and discussion well-written"
#   - p-hacking/opt stopping:  "p-value reported, grammar correct"
#   - Underpowered null:       "sample size stated, conclusion phrased clearly"
#   - Measurement invariance:  "groups described, tables labelled"
#   - Common method variance:  "correlation reported, survey described"
#   - NHST equivalence error:  "conclusion stated clearly"   (clinically dangerous)
#
# Baseline LLM output on hard examples:  "grammar correct, p-values reported"
# LLM-as-judge score for that output:    0.04–0.08
# Expected baseline holdout score:       ~0.05–0.15
#
# After evolution the skill gains research-methods guidance:
#   pre-registration, HARKing, p-hacking, statistical_power, effect_size,
#   confidence_intervals, regression_to_mean, measurement_invariance,
#   common_method_variance, equivalence_testing, omitted_variable_bias.
# Expected evolved holdout score:        ~0.55–0.75
# ══════════════════════════════════════════════════════════════════════════════
from .easy   import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard   import GOLDEN_EXAMPLES_HARD

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
