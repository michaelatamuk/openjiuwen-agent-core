# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Import session logs from a local Hermes agent installation.

Reads ~/.hermes/sessions/*.json — the same format as JiuwenSessionImporter
(both write {session_id, messages: [{role, content}]} JSON files) but at the
Hermes default path instead of the Jiuwen default path.

Useful when evolving skills offline on a machine that also runs hermes-agent,
allowing the Hermes session history to contribute to the eval dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .secret_patterns import _contains_secret


class HermesSessionImporter:
    """Import user/assistant pairs from ~/.hermes/sessions/*.json"""

    SESSION_DIR = Path.home() / ".hermes" / "sessions"

    @staticmethod
    def extract_messages(limit: int = 0) -> List[Dict]:
        if not HermesSessionImporter.SESSION_DIR.exists():
            return []
        messages = []
        session_files = sorted(
            HermesSessionImporter.SESSION_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for sf in session_files:
            try:
                data = json.loads(sf.read_text(encoding="utf-8"))
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
                        "source": "hermes",
                        "task_input": user_text,
                        "assistant_response": assistant_text,
                        "session_id": session_id,
                    }
                )
                if limit and len(messages) >= limit:
                    return messages
        return messages
