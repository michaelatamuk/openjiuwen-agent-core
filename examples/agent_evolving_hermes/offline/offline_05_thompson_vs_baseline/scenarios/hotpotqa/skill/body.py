# coding: utf-8
# Baseline skill — deliberately shallow.
# Answers questions directly from memory without chaining facts.
# The expected_behavior requires the skill to show each reasoning hop
# explicitly before stating the answer.
#
# Expected baseline score  : ~0.15–0.30  (answer word sometimes present)
# Expected evolved score   : ~0.55–0.75  (explicit hop chain + answer)

SKILL_BODY = """\
You are a question-answering assistant. When given a question, answer it
directly and concisely in one or two sentences.

State only the final answer.  Do not explain how you arrived at it or
list intermediate facts that led to the answer.
"""
