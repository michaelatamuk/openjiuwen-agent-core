from __future__ import annotations

from typing import Tuple

from openjiuwen.agent_evolving_hermess.online.background_review_prompts import select_prompt
from openjiuwen.agent_evolving_hermess.online.types import ReviewTrigger

_SYSTEM_PROMPT = (
    "You are a background review agent. Your only job is to analyze the "
    "conversation provided and make targeted updates to skills and/or memory "
    "using the tools available. You must NOT do anything else. "
    "Do not explain your reasoning at length — just call the tools."
)


def build_review_prompts(trigger: ReviewTrigger) -> Tuple[str, str]:
    """Select the review prompt for the trigger mode and return the system prompt.

    Returns (review_prompt, system_prompt).
    """
    review_prompt = select_prompt(trigger.mode)
    return review_prompt, _SYSTEM_PROMPT
