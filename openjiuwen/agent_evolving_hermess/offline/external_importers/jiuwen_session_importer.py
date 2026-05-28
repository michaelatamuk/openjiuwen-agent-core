# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Session log mining for eval dataset construction.

Mirrors hermes-agent-self-evolution evolution/core/external_importers.py.
Key Jiuwen adaptation: JiuwenSessionImporter reads ~/.jiuwen/sessions/*.json
(same format as HermesSessionImporter but different path).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from .secret_patterns import _contains_secret

# ── Jiuwen session importer ───────────────────────────────────────────────────


class JiuwenSessionImporter:
    """Import user/assistant pairs from ~/.jiuwen/sessions/*.json"""

    SESSION_DIR = Path.home() / ".jiuwen" / "sessions"

    @staticmethod
    def extract_messages(limit: int = 0) -> List[Dict]:
        if not JiuwenSessionImporter.SESSION_DIR.exists():
            return []
        messages = []
        session_files = sorted(
            JiuwenSessionImporter.SESSION_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for sf in session_files:
            try:
                data = json.loads(sf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            msg_list = data.get("messages", [])
            session_id = data.get("session_id", sf.stem)
            for i, msg in enumerate(msg_list):
                if msg.get("role") != "user":
                    continue
                user_text = msg.get("content", "")
                if not user_text or len(user_text) < 10:
                    continue
                if _contains_secret(user_text):
                    continue
                assistant_text = ""
                for j in range(i + 1, len(msg_list)):
                    if msg_list[j].get("role") == "assistant":
                        c = msg_list[j].get("content", "")
                        if c:
                            assistant_text = c
                            break
                    elif msg_list[j].get("role") == "user":
                        break
                if assistant_text and _contains_secret(assistant_text):
                    continue
                messages.append(
                    {
                        "source": "jiuwen",
                        "task_input": user_text,
                        "assistant_response": assistant_text,
                        "session_id": session_id,
                    }
                )
                if limit and len(messages) >= limit:
                    return messages
        return messages
