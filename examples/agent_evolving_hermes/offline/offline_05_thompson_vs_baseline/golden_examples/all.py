# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 12 hand-crafted examples: 4 easy · 4 medium · 4 hard
#
# Design rationale: easy examples are visible even to a shallow skill.
# Medium and hard examples (security bugs, concurrency, N+1) only surface in
# a genuinely deep review.  TS will learn to focus budget on these hard
# examples, which are most discriminating between evolved variants.
# ══════════════════════════════════════════════════════════════════════════════
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.golden_examples.easy import \
    GOLDEN_EXAMPLES_EASY
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.golden_examples.hard import \
    GOLDEN_EXAMPLES_HARD
from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.golden_examples.medium import \
    GOLDEN_EXAMPLES_MEDIUM

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
