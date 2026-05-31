# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 16 hand-crafted examples: 4 easy · 4 medium · 8 hard
#
# Design rationale — why this scenario produces a VERY LOW baseline score
# ────────────────────────────────────────────────────────────────────────
# Baseline skill: structural / completeness checker only.
#   Explicitly instructed NOT to apply any legal doctrine or assess
#   enforceability of individual clauses.
#
# Hard legal traps are INVISIBLE to a structural reviewer:
#   - Liquidated damages penalty:    "payment terms present and specified"
#   - Force majeure ejusdem generis: "force majeure clause included"
#   - Uncapped IP indemnification:   "indemnification clause present"
#   - Anti-assignment / CoC gap:     "assignment restriction present"
#   - Auto-renewal notice trap:      "renewal terms are stated"
#   - UCC warranty disclaimer gap:   "warranty limitation included"
#   - Overbroad non-compete:         "non-compete clause present"
#   - One-sided indemnification:     "indemnification terms defined"
#
# Baseline LLM output on hard examples:  "clause present, terms seem clear"
# LLM-as-judge score for that output:    0.04–0.08
# Expected baseline holdout score:       ~0.05–0.15
#
# After evolution the skill gains contract law guidance:
#   liquidated_damages, penalty_clause, ejusdem_generis, force_majeure,
#   IP_indemnification, aggregate_liability_cap, change_of_control,
#   auto-renewal, warranty_disclaimer, merchantability, non-compete,
#   blue-penciling, comparative_fault, mutual_indemnification.
# Expected evolved holdout score:        ~0.50–0.70
# ══════════════════════════════════════════════════════════════════════════════
from .easy   import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard   import GOLDEN_EXAMPLES_HARD

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
