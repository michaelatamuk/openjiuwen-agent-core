from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from openjiuwen.agent_evolving_hermess.online.config import BackgroundReviewConfig
from openjiuwen.agent_evolving_hermess.online.memory_store import MemoryStore


def build_conversation_context(
    messages_snapshot: List[Dict[str, Any]],
    config: BackgroundReviewConfig,
) -> Tuple[Path, MemoryStore, str]:
    """Resolve storage roots, create MemoryStore, serialise conversation to text.

    Returns (skills_root, memory_store, conversation_text).
    """
    skills_root = config.skills_root or (Path.home() / ".jiuwen" / "skills")
    memory_root = config.memory_root or (Path.home() / ".jiuwen" / "memories")
    memory_store = MemoryStore(
        memory_root=memory_root,
        memory_char_limit=config.memory_char_limit,
        user_char_limit=config.user_char_limit,
    )
    conversation_text = messages_to_text(messages_snapshot)
    return skills_root, memory_store, conversation_text


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
