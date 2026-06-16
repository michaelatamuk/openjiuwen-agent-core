# ── Feature definition ─────────────────────────────────────────────────────────

# Only columns available BEFORE any GEPA run (a-priori).
# Discrimination stats (discrimination_mean / cv / ...) and cross_metric_corr_mean
# are computed from the GEPA call log and are therefore excluded.
NUM_COLS = [
    "n_examples_trainset",
    "n_examples_valset",
    "n_examples_holdout",
    "baseline_skill_chars",
    "baseline_skill_body_chars",
]
CAT_METRIC = ["metric"]
CAT_MODEL  = ["model"]
TEXT_COL   = "description"
TARGET     = "improvement"
GROUP_COL  = "skill_name"

ALL_FEATURE_COLS = NUM_COLS + CAT_METRIC + CAT_MODEL + [TEXT_COL]
