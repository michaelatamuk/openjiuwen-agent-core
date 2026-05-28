# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Background review prompt texts.

Faithful translations of Hermess's three review prompts into the Jiuwen context.
"""
from __future__ import annotations

from openjiuwen.agent_evolving_hermess.online.background_review_prompts.combined import COMBINED_REVIEW_PROMPT
from openjiuwen.agent_evolving_hermess.online.background_review_prompts.memory import MEMORY_REVIEW_PROMPT
from openjiuwen.agent_evolving_hermess.online.background_review_prompts.skill import SKILL_REVIEW_PROMPT
from openjiuwen.agent_evolving_hermess.online.types import ReviewMode


def select_prompt(mode: "ReviewMode") -> str:  # noqa: F821
    """Return the correct prompt string for the given ReviewMode."""
    if mode == ReviewMode.MEMORY_ONLY:
        return MEMORY_REVIEW_PROMPT
    if mode == ReviewMode.SKILLS_ONLY:
        return SKILL_REVIEW_PROMPT
    return COMBINED_REVIEW_PROMPT
