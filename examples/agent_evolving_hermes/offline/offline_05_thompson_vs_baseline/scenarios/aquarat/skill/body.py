# coding: utf-8
# Baseline skill — deliberately shallow.
# It identifies the problem type and estimates a rough answer without
# performing any algebraic calculation or selecting the correct option.
#
# GEPA must discover that the expected behaviour requires:
#   1. Setting up the algebraic equation explicitly
#   2. Solving it step-by-step showing all arithmetic
#   3. Matching the computed result to the correct lettered option
#   4. Stating the option letter clearly (e.g. "Answer: C")
#
# Expected baseline score  : ~0.05–0.15  (no option selected; estimation off)
# Expected evolved score   : ~0.55–0.75  (full working + correct option letter)

SKILL_BODY = """\
You are a math assistant. When given an algebra word problem with multiple
choice options, read the question and identify what type of calculation is
involved (e.g. percentage, ratio, rate, mixture).

Provide a rough estimate of the answer based on the numbers in the question
and indicate which option seems closest to your estimate.
"""
