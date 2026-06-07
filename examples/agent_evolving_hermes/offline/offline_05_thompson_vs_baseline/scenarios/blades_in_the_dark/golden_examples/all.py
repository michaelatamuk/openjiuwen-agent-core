# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 16 hand-crafted examples: 4 easy · 4 medium · 8 hard
#
# Scenario: blades-in-the-dark  —  Blades in the Dark TTRPG GM facilitation
#
# Baseline skill: generic D&D 5e-style TTRPG facilitator
#
# WHY this produces a VERY LOW baseline score (~0.05–0.15)
# ──────────────────────────────────────────────────────────
# The D&D baseline skill primes the LLM to apply d20/DC/HP/rest mechanics.
# Hard examples test BitD-exclusive mechanics with no D&D equivalent, where
# the D&D framework gives systematically wrong answers:
#
#   Flashback:           D&D says "too late to plan" → BitD formal Flashback mechanic
#   Engagement Roll:     D&D skips straight to scene → BitD mandatory pre-score roll
#   Devil's Bargain:     D&D Inspiration (no downside) → BitD complication regardless of result
#   Harm (no HP):        D&D HP + death saves → BitD Harm levels, no numbers
#   Heat / Entanglements:D&D abstract reputation → BitD numerical post-score pressure
#   Fortune Roll:        D&D skill check vs DC → BitD no-DC GM-only roll
#   Vice / Stress:       D&D Long Rest → BitD explicit Downtime activity required
#   Trauma (not death):  D&D 0 HP / unconscious → BitD permanent condition, play continues
#
# Expected baseline holdout score:  ~0.05–0.15
# Expected evolved holdout score:   ~0.60–0.75
# ══════════════════════════════════════════════════════════════════════════════

from .easy   import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard   import GOLDEN_EXAMPLES_HARD

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
