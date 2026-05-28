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
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openjiuwen.agent_evolving_hermess.background_review_prompts import select_prompt
from openjiuwen.agent_evolving_hermess.config import BackgroundReviewConfig
from openjiuwen.agent_evolving_hermess.memory_store import MemoryStore
from openjiuwen.agent_evolving_hermess.provenance import make_write_metadata  # noqa: F401
from openjiuwen.agent_evolving_hermess.skill_store import (
    skill_create,
    skill_edit,
    skill_list,  # noqa: F401
    skill_patch,
)
from openjiuwen.agent_evolving_hermess.types import (
    ReviewAction,
    ReviewMode,  # noqa: F401
    ReviewResult,
    ReviewTrigger,
)

logger = logging.getLogger(__name__)

# ── Tool schemas exposed to the review LLM ───────────────────────────────────

SKILL_WRITE_TOOL = {
    "name": "skill_write",
    "description": (
        "Create a new skill or fully replace an existing skill's SKILL.md. "
        "Use skill_patch for small targeted changes. "
        "The content must begin with YAML frontmatter (--- name: ... description: ... ---)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "edit"],
                "description": "'create' for new skills, 'edit' to replace an existing skill.",
            },
            "name": {
                "type": "string",
                "description": "Skill directory name (lowercase, hyphens ok).",
            },
            "content": {
                "type": "string",
                "description": "Full SKILL.md text including frontmatter.",
            },
            "category": {
                "type": "string",
                "description": "Optional category subdirectory (only used with action='create').",
            },
        },
        "required": ["action", "name", "content"],
    },
}

SKILL_PATCH_TOOL = {
    "name": "skill_patch",
    "description": (
        "Apply a targeted string-replacement patch to an existing SKILL.md. "
        "Prefer this over skill_write for small targeted changes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "old_string": {
                "type": "string",
                "description": "Exact text to find and replace.",
            },
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "default": False,
                "description": "Replace all occurrences (default: first only).",
            },
        },
        "required": ["name", "old_string", "new_string"],
    },
}

MEMORY_WRITE_TOOL = {
    "name": "memory_write",
    "description": "Add, replace, or remove an entry in the agent memory or user profile store.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "replace", "remove"]},
            "target": {"type": "string", "enum": ["memory", "user"]},
            "content": {
                "type": "string",
                "description": "New content (required for add/replace).",
            },
            "old_text": {
                "type": "string",
                "description": "Text to find (required for replace/remove).",
            },
        },
        "required": ["action", "target"],
    },
}

REVIEW_TOOLS = [SKILL_WRITE_TOOL, SKILL_PATCH_TOOL, MEMORY_WRITE_TOOL]

# ── Message serialization ─────────────────────────────────────────────────────


def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
    """Convert conversation snapshot to plain text for the review prompt."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle tool-result message content blocks
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", block.get("content", "")))
                else:
                    parts.append(str(block))
            content = " ".join(parts)
        if content:
            lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


# ── LLM call ─────────────────────────────────────────────────────────────────


async def _call_review_llm(
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


# ── Tool dispatch ─────────────────────────────────────────────────────────────


async def _dispatch_tool_call(
    tool_name: str,
    args: Dict[str, Any],
    skill_store_root: Path,
    memory_store: MemoryStore,
    config: BackgroundReviewConfig,
    session_id: str,
) -> Tuple[bool, str, Optional[ReviewAction]]:
    """Execute one tool call. Returns (success, message, action_or_none)."""

    if tool_name == "skill_write":
        action_type = args.get("action", "edit")
        name = args.get("name", "")
        content = args.get("content", "")
        category = args.get("category")
        if action_type == "create":
            ok, msg = await skill_create(
                name,
                content,
                skill_store_root,
                category=category,
                protected_names=config.protected_skill_names,
            )
        else:
            ok, msg = await skill_edit(
                name,
                content,
                skill_store_root,
                protected_names=config.protected_skill_names,
            )
        if ok:
            return True, msg, ReviewAction(
                action_type=f"skill_{action_type}",
                target_name=name,
                summary=f"{'Created' if action_type == 'create' else 'Edited'} skill '{name}'",
                session_id=session_id,
            )
        return False, msg, None

    if tool_name == "skill_patch":
        name = args.get("name", "")
        old_s = args.get("old_string", "")
        new_s = args.get("new_string", "")
        replace_all = args.get("replace_all", False)
        ok, msg = await skill_patch(
            name,
            old_s,
            new_s,
            skill_store_root,
            replace_all=replace_all,
            protected_names=config.protected_skill_names,
        )
        if ok:
            return True, msg, ReviewAction(
                action_type="skill_patch",
                target_name=name,
                summary=f"Patched skill '{name}'",
                session_id=session_id,
            )
        return False, msg, None

    if tool_name == "memory_write":
        mem_action = args.get("action", "add")
        target = args.get("target", "memory")
        content = args.get("content", "")
        old_text = args.get("old_text", "")
        if mem_action == "add":
            ok, msg = memory_store.add(target, content)
        elif mem_action == "replace":
            ok, msg = memory_store.replace(target, old_text, content)
        elif mem_action == "remove":
            ok, msg = memory_store.remove(target, old_text)
        else:
            return False, f"Unknown memory action '{mem_action}'", None
        if ok:
            return True, msg, ReviewAction(
                action_type=f"memory_{mem_action}",
                target_name=target,
                summary=f"{'Memory' if target == 'memory' else 'User profile'} {mem_action}",
                session_id=session_id,
            )
        return False, msg, None

    return False, f"Unknown tool '{tool_name}'", None


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

    conversation_text = _messages_to_text(messages_snapshot)
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
            _call_review_llm(
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
        ok, msg, action = await _dispatch_tool_call(
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
