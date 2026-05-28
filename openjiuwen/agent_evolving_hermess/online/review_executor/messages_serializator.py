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

from typing import Any, Dict, List, Optional


# ── Message serialization ─────────────────────────────────────────────────────


def messages_to_text(messages: List[Dict[str, Any]]) -> str:
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
