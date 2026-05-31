# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 20 hand-crafted examples: 4 easy · 8 medium · 8 hard
#
# Design rationale — why this scenario produces a VERY LOW baseline score
# ────────────────────────────────────────────────────────────────────────
# The baseline skill says only "check for memory leaks, buffer overflows,
# error handling, code quality" — no mention of RTOS, ISR, FreeRTOS, volatile.
#
# Hard/medium RTOS bugs are INVISIBLE to a generic C reviewer:
#   - malloc() in ISR:       "memory allocation looks correct"  (wrong: deadlock)
#   - volatile missing:      "variable declarations look fine"  (wrong: inf loop)
#   - Mutex in ISR:          "synchronization looks correct"    (wrong: HardFault)
#   - xQueueSend portMAX_DELAY in ISR:  "queue usage fine"     (wrong: UB/assert)
#   - portYIELD_FROM_ISR missing:       "semaphore usage ok"   (wrong: latency)
#   - 64-bit torn MMIO read: "looks fine"                       (wrong: data race)
#
# Baseline LLM output on hard examples:  "looks fine, add error handling"
# LLM-as-judge score for that output:    0.05–0.10
# Expected baseline holdout score:       ~0.10–0.20
#
# After evolution the skill gains RTOS-specific guidance:
#   volatile on ISR-shared variables, *FromISR() variants, portYIELD_FROM_ISR,
#   pvPortMalloc vs malloc, __disable_irq() for atomic reads, __DSB() barriers.
# Expected evolved holdout score:        ~0.60–0.75
# ══════════════════════════════════════════════════════════════════════════════
from .easy   import GOLDEN_EXAMPLES_EASY
from .medium import GOLDEN_EXAMPLES_MEDIUM
from .hard   import GOLDEN_EXAMPLES_HARD

GOLDEN_EXAMPLES = GOLDEN_EXAMPLES_EASY + GOLDEN_EXAMPLES_MEDIUM + GOLDEN_EXAMPLES_HARD
