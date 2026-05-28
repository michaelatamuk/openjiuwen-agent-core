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

import json

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from openjiuwen.agent_evolving_hermess.online.config import BackgroundReviewConfig
from openjiuwen.agent_evolving_hermess.online.review_executor.tools._review_tools import REVIEW_TOOLS
from openjiuwen.core.common.logging import logger


async def call_llm_with_timeout(
    system_prompt: str,
    conversation_text: str,
    review_prompt: str,
    model: str,
    config: BackgroundReviewConfig,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Call the review LLM with a hard timeout, absorbing errors into the return value.

    Returns (tool_calls, error) where error is None on success.
    On timeout or exception, returns ([], error_message).
    """
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
        return tool_calls, None
    except asyncio.TimeoutError:
        error = f"Background review timed out after {config.review_timeout_seconds}s"
        logger.warning(error)
        return [], error
    except Exception as e:
        error = f"Background review LLM call failed: {e}"
        logger.warning(error)
        return [], error


async def call_review_llm(
    system_prompt: str,
    conversation_text: str,
    review_prompt: str,
    model: str,
    max_iterations: int,
) -> List[Dict[str, Any]]:
    """Run a tool-enabled LLM conversation for background review.

    Uses litellm (same dependency Jiuwen uses for LLM calls in other places).
    Returns list of tool call dicts: [{"tool": name, "args": {...}}, ...]
    """
    try:
        import litellm
    except ImportError:
        logger.error("litellm not installed; cannot run background review LLM call.")
        return []

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation_text},
        {"role": "user", "content": review_prompt},
    ]
    tool_calls_made = []
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                tools=[{"type": "function", "function": t} for t in REVIEW_TOOLS],
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            logger.warning("BackgroundReview LLM call failed: %s", e)
            break

        choice = response.choices[0]
        message = choice.message

        # Collect any tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls_made.append(
                    {"tool": tc.function.name, "id": tc.id, "args": args}
                )
                # Append assistant + synthetic tool result to continue the loop
                messages.append(
                    {"role": "assistant", "content": None, "tool_calls": [tc]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"queued": True}),
                    }
                )
        else:
            # No more tool calls — model is done
            break

        if choice.finish_reason == "stop":
            break

    return tool_calls_made
