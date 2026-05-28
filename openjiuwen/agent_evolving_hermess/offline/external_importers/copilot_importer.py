# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Import conversation history from GitHub Copilot Chat (VSCode extension).

Mirrors hermes-agent-self-evolution evolution/core/external_importers.py
CopilotImporter — reads VSCode Copilot Chat session files across platforms.

Storage paths searched (newest first by mtime):
  macOS:   ~/Library/Application Support/Code/User/globalStorage/github.copilot-chat/
  Linux:   ~/.config/Code/User/globalStorage/github.copilot-chat/
  Windows: %APPDATA%/Code/User/globalStorage/github.copilot-chat/
  All:     ~/Library/Application Support/Code - Insiders/… (VSCode Insiders)

Both `.json` and `.jsonl` files are parsed.  The importer recognises two
conversation shapes produced by different Copilot Chat versions:

  Shape A (array of turns):
    [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}, …]

  Shape B (session object with turns list):
    {"turns": [{"role": "user", "content": "…"}, …]}

  Shape C (JSONL — one JSON object per line):
    {"role": "user", "content": "…"}
    {"role": "assistant", "content": "…"}
"""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional

from .secret_patterns import _contains_secret


def _vscode_storage_dirs() -> List[Path]:
    """Return candidate VSCode user-data directories for the current platform."""
    home = Path.home()
    system = platform.system()

    candidates: List[Path] = []
    if system == "Darwin":
        candidates = [
            home / "Library" / "Application Support" / "Code" / "User",
            home / "Library" / "Application Support" / "Code - Insiders" / "User",
            home / "Library" / "Application Support" / "VSCodium" / "User",
        ]
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            candidates = [
                Path(appdata) / "Code" / "User",
                Path(appdata) / "Code - Insiders" / "User",
            ]
    else:  # Linux / BSD
        xdg = os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))
        candidates = [
            Path(xdg) / "Code" / "User",
            Path(xdg) / "Code - Insiders" / "User",
            Path(xdg) / "VSCodium" / "User",
        ]
    return [d for d in candidates if d.exists()]


def _chat_files(vscode_user_dir: Path) -> List[Path]:
    """Collect all Copilot Chat data files under a VSCode user directory."""
    glob_roots = [
        vscode_user_dir / "globalStorage" / "github.copilot-chat",
        vscode_user_dir / "globalStorage" / "GitHub.copilot-chat",
    ]
    workspace_root = vscode_user_dir / "workspaceStorage"

    files: List[Path] = []
    for root in glob_roots:
        if root.exists():
            files.extend(root.rglob("*.json"))
            files.extend(root.rglob("*.jsonl"))

    # Workspace-scoped storage (older Copilot versions)
    if workspace_root.exists():
        for ws_dir in workspace_root.iterdir():
            if not ws_dir.is_dir():
                continue
            for subdir_name in ("GitHub.copilot-chat", "github.copilot-chat"):
                subdir = ws_dir / subdir_name
                if subdir.exists():
                    files.extend(subdir.rglob("*.json"))
                    files.extend(subdir.rglob("*.jsonl"))

    return sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)


def _extract_turns(data) -> List[Dict]:
    """Extract a flat list of {role, content} dicts from various JSON shapes."""
    # Shape B: {"turns": [...]}
    if isinstance(data, dict) and "turns" in data:
        data = data["turns"]

    # Shape A: array of turn dicts
    if isinstance(data, list):
        turns = []
        for item in data:
            if isinstance(item, dict) and "role" in item and "content" in item:
                turns.append({"role": item["role"], "content": item["content"]})
            # Some versions nest turns inside a "messages" key inside the item
            elif isinstance(item, dict) and "messages" in item:
                for m in (item["messages"] or []):
                    if isinstance(m, dict) and "role" in m and "content" in m:
                        turns.append({"role": m["role"], "content": m["content"]})
        return turns

    return []


def _parse_file(path: Path) -> List[Dict]:
    """Parse one Copilot Chat file and return a flat list of message turns."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []

    # Try JSONL first (line-delimited JSON)
    if path.suffix == ".jsonl":
        turns: List[Dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "role" in obj and "content" in obj:
                    turns.append({"role": obj["role"], "content": obj["content"]})
            except json.JSONDecodeError:
                continue
        return turns

    # Try JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    # The file might be a list of session objects
    if isinstance(data, list) and data and isinstance(data[0], dict):
        # Check if it looks like a sessions list
        if "turns" in data[0] or ("messages" in data[0] and isinstance(data[0]["messages"], list)):
            all_turns: List[Dict] = []
            for session in data:
                all_turns.extend(_extract_turns(session))
            return all_turns

    return _extract_turns(data)


class CopilotImporter:
    """Import conversation pairs from GitHub Copilot Chat VSCode extension storage."""

    @staticmethod
    def extract_messages(limit: int = 0) -> List[Dict]:
        """Return list of {source, task_input, assistant_response} dicts.

        Args:
            limit: Maximum number of message pairs to return (0 = unlimited).
        """
        all_files: List[Path] = []
        for vscode_dir in _vscode_storage_dirs():
            all_files.extend(_chat_files(vscode_dir))

        if not all_files:
            return []

        messages: List[Dict] = []
        for chat_file in all_files:
            turns = _parse_file(chat_file)
            # Walk turns looking for user → assistant pairs
            for i, turn in enumerate(turns):
                if turn.get("role") != "user":
                    continue
                user_text = turn.get("content", "")
                if not user_text or not isinstance(user_text, str) or len(user_text) < 10:
                    continue
                if _contains_secret(user_text):
                    continue
                assistant_text = ""
                for j in range(i + 1, len(turns)):
                    if turns[j].get("role") == "assistant":
                        c = turns[j].get("content", "")
                        if c and isinstance(c, str):
                            assistant_text = c
                        break
                    if turns[j].get("role") == "user":
                        break
                if assistant_text and _contains_secret(assistant_text):
                    continue
                messages.append(
                    {
                        "source": "copilot",
                        "task_input": user_text,
                        "assistant_response": assistant_text,
                    }
                )
                if limit and len(messages) >= limit:
                    return messages
        return messages
