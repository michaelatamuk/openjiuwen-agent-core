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

# ── Claude Code importer (same as Hermes) ────────────────────────────────────


class ClaudeCodeImporter:
    HISTORY_PATH = Path.home() / ".claude" / "history.jsonl"

    @staticmethod
    def extract_messages(limit: int = 0) -> List[Dict]:
        if not ClaudeCodeImporter.HISTORY_PATH.exists():
            return []
        messages = []
        with open(ClaudeCodeImporter.HISTORY_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = entry.get("display", "")
                if not text or len(text) < 10 or _contains_secret(text):
                    continue
                messages.append(
                    {
                        "source": "claude-code",
                        "task_input": text,
                        "session_id": entry.get("sessionId", ""),
                    }
                )
                if limit and len(messages) >= limit:
                    break
        return messages
