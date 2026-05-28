# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Core background review execution.

Hermess pattern: forks AIAgent, restricts to skill_manage + memory tools,
runs review prompt over conversation snapshot, parses tool call outputs
into ReviewAction list.

Jiuwen mapping:
  - No forked AIAgent available; instead, make a direct LLM call with
    the conversation + review prompt and parse the JSON tool-call outputs.
  - Uses litellm (same dependency Jiuwen uses for other LLM calls).
  - Tool definitions for skill_write, skill_patch, skill_create,
    memory_write are defined here as JSON schemas and validated locally.
  - Actual writes are executed by skill_store.py and memory_store.py.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openjiuwen.agent_evolving_hermess.online.review_executor._messages_serializator import messages_to_text
from openjiuwen.agent_evolving_hermess.online.review_executor.tool_call_dispatcher import dispatch_tool_call
from openjiuwen.core.common.logging import logger
from openjiuwen.agent_evolving_hermess.online.review_executor.review_llm_caller import call_review_llm
from openjiuwen.agent_evolving_hermess.online.background_review_prompts import select_prompt
from openjiuwen.agent_evolving_hermess.online.config import BackgroundReviewConfig
from openjiuwen.agent_evolving_hermess.online.memory_store import MemoryStore
from openjiuwen.agent_evolving_hermess.online.types import ReviewAction, ReviewResult, ReviewTrigger



# ── Main entry point ──────────────────────────────────────────────────────────


async def run_background_review(
    messages_snapshot: List[Dict[str, Any]],
    trigger: ReviewTrigger,
    config: BackgroundReviewConfig,
    model: str,
    session_id: str,
    parent_session_id: str = "",
) -> ReviewResult:
    """Core review execution. Mirrors Hermess _run_review_in_thread().

    1. Build conversation text from snapshot.
    2. Select review prompt by mode.
    3. Call LLM with tool definitions (skill_write, skill_patch, memory_write).
    4. Dispatch each tool call to skill_store / memory_store.
    5. Return ReviewResult with all actions taken.
    """
    skills_root = config.skills_root or (Path.home() / ".jiuwen" / "skills")
    memory_root = config.memory_root or (Path.home() / ".jiuwen" / "memories")
    memory_store = MemoryStore(
        memory_root=memory_root,
        memory_char_limit=config.memory_char_limit,
        user_char_limit=config.user_char_limit,
    )

    conversation_text = messages_to_text(messages_snapshot)
    review_prompt = select_prompt(trigger.mode)

    system_prompt = (
        "You are a background review agent. Your only job is to analyze the "
        "conversation provided and make targeted updates to skills and/or memory "
        "using the tools available. You must NOT do anything else. "
        "Do not explain your reasoning at length — just call the tools."
    )

    t0 = time.monotonic()
    actions: List[ReviewAction] = []
    error: Optional[str] = None

    try:
        tool_calls = await asyncio.wait_for(
            call_review_llm(
                system_prompt=system_prompt,
                conversation_text=conversation_text,
                review_prompt=review_prompt,
                model=model,
                max_iterations=config.review_max_iterations,
            ),
            timeout=config.review_timeout_seconds,
        )
    except asyncio.TimeoutError:
        error = f"Background review timed out after {config.review_timeout_seconds}s"
        logger.warning(error)
        tool_calls = []
    except Exception as e:
        error = f"Background review LLM call failed: {e}"
        logger.warning(error)
        tool_calls = []

    for tc in tool_calls:
        ok, msg, action = await dispatch_tool_call(
            tool_name=tc["tool"],
            args=tc["args"],
            skill_store_root=skills_root,
            memory_store=memory_store,
            config=config,
            session_id=session_id,
        )
        if ok and action:
            actions.append(action)
        elif not ok:
            logger.debug(
                "BackgroundReview tool call failed: %s — %s", tc["tool"], msg
            )

    duration = time.monotonic() - t0
    summary = " · ".join(a.summary for a in actions) if actions else "No changes"

    return ReviewResult(
        trigger=trigger,
        actions=actions,
        error=error,
        duration_seconds=duration,
        summary_line=summary,
    )
