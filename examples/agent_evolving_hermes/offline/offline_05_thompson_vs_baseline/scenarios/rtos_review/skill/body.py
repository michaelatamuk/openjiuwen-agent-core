# coding: utf-8
# ─────────────────────────────────────────────────────────────────────────────
# DELIBERATELY SHALLOW BASELINE — generic C code review, no RTOS knowledge
# This skill will score very low on hard RTOS examples because:
#   - malloc() in ISR:  reviewer says "check NULL" (misses heap-lock deadlock)
#   - volatile missing: reviewer says "looks fine" (misses compiler caching)
#   - Mutex in ISR:     reviewer says "synchronization looks correct" (wrong)
#   - xQueueSend() blocking in ISR: reviewer says "queue usage fine" (UB)
#   - portYIELD_FROM_ISR missing: reviewer says "semaphore usage ok" (latency)
# ─────────────────────────────────────────────────────────────────────────────
SKILL_BODY = """\
When reviewing C code, check for:

1. **Memory safety** — memory leaks, double-free, use-after-free; always check
   the return value of `malloc()` / `calloc()` for NULL before use.

2. **Buffer safety** — buffer overflows, off-by-one errors; replace unsafe
   functions like `strcpy` and `sprintf` with bounded alternatives
   (`strncpy`, `snprintf`).

3. **Error handling** — every function that can fail should have its return
   value checked; missing error paths hide bugs in production.

4. **Code quality** — avoid magic numbers (use named `#define` constants),
   keep functions short and single-purpose, use descriptive variable names.

5. **Undefined behaviour** — uninitialized variables, signed/unsigned
   comparison mismatches, integer overflow on size calculations.

For each issue found, state what is wrong, why it matters, and show a
corrected code snippet.
"""
