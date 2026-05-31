# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 20 hand-crafted examples: 4 easy · 8 medium · 8 hard
#
# Design rationale — why this scenario produces a VERY LOW baseline score
# ────────────────────────────────────────────────────────────────────────
# Baseline skill: "evaluate clarity, logical flow, completeness, and relevance"
# — standard writing-quality review, no statistical methods knowledge.
#
# Hard methodology issues are INVISIBLE to a generic writing reviewer:
#   - HARKing:                 "findings clearly stated"         (misses fraud)
#   - p-hacking/opt stopping:  "methods well described"          (misses Type I inflation)
#   - Underpowered null:       "conclusions follow from data"    (misses Type II error)
#   - Measurement invariance:  "groups well-characterised"       (misses invalid comparison)
#   - Common method variance:  "constructs clearly measured"     (misses inflated r)
#   - NHST equivalence error:  "conclusions are logical"         (clinically dangerous)
#
# Baseline LLM output on hard examples:  "well-written, findings clearly presented"
# LLM-as-judge score for that output:    0.04–0.10
# Expected baseline holdout score:       ~0.10–0.20
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
