# ══════════════════════════════════════════════════════════════════════════════
# BASELINE SKILL — deliberately generic with NO machine-learning knowledge.
#
# WHY this produces a LOW baseline score
# ──────────────────────────────────────
# Data-leakage bugs look superficially correct to a generic code reviewer:
#   - `scaler.fit_transform(X)` before the split LOOKS fine — it's valid Python
#   - `KFold` on patient rows LOOKS fine — cross-validation is "best practice"
#   - SMOTE before split LOOKS fine — the class balance is "fixed"
#
# Without ML-specific guidance the LLM says "looks good" or "add comments",
# and the LLM-as-judge gives 0.05–0.20 for hard examples.  An evolved skill
# that explicitly names leakage patterns pushes scores to 0.70–0.90.
#
# What the baseline deliberately OMITS:
#   - data_leakage / fit before split
#   - Pipeline to prevent preprocessing leakage
#   - GroupKFold / TimeSeriesSplit for structured data
#   - SMOTE-inside-fold vs SMOTE-before-split
#   - Target encoding leakage
#   - Nested cross-validation bias
#   - stratify for imbalanced splits
#   - random_state for reproducibility
#   - appropriate evaluation metric for imbalanced classes
# ══════════════════════════════════════════════════════════════════════════════

SKILL_BODY = """\
# Code Review Assistant

When reviewing code, look for the following general issues:

- Check for bugs and logical errors in the code
- Look for performance problems
- Identify code quality issues such as missing error handling
- Suggest improvements where applicable

Provide your findings in a clear format.
"""
