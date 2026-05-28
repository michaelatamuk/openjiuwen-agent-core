# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Simple flat-file memory store.

Mirrors Hermess MemoryStore exactly.
Two stores: "memory" (agent observations) and "user" (user profile).
Uses fcntl file locking and atomic writes.
"""
from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path
from typing import IO, List, Optional, Tuple

_ENTRY_DELIMITER = "\n§\n"


def _default_memory_root() -> Path:
    return Path.home() / ".jiuwen" / "memories"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_entries(path: Path) -> List[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    return [e.strip() for e in raw.split(_ENTRY_DELIMITER) if e.strip()]


def _write_entries(path: Path, entries: List[str]) -> None:
    _atomic_write(path, _ENTRY_DELIMITER.join(entries))


class MemoryStore:
    """File-backed key-value memory. Mirrors Hermess MemoryStore."""

    def __init__(
        self,
        memory_root: Optional[Path] = None,
        memory_char_limit: int = 2200,
        user_char_limit: int = 1375,
    ):
        self._root = memory_root or _default_memory_root()
        self._limits = {"memory": memory_char_limit, "user": user_char_limit}
        self._paths = {
            "memory": self._root / "MEMORY.md",
            "user": self._root / "USER.md",
        }
        self._lock_paths = {
            "memory": self._root / "MEMORY.md.lock",
            "user": self._root / "USER.md.lock",
        }

    def _acquire(self, target: str) -> IO:
        lock_path = self._lock_paths[target]
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "w")
        fcntl.flock(fh, fcntl.LOCK_EX)
        return fh

    def _release(self, fh: IO) -> None:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()

    def add(self, target: str, content: str) -> Tuple[bool, str]:
        content = content.strip()
        if not content:
            return False, "Content must not be empty."
        if target not in self._paths:
            return False, f"Unknown target '{target}'. Use 'memory' or 'user'."

        fh = self._acquire(target)
        try:
            entries = _read_entries(self._paths[target])
            if content in entries:
                return False, "Exact duplicate — already stored."
            new_total = (
                sum(len(e) for e in entries)
                + len(content)
                + len(_ENTRY_DELIMITER)
            )
            if new_total > self._limits[target]:
                return False, (
                    f"Would exceed {target} limit of {self._limits[target]} chars. "
                    f"Current: {sum(len(e) for e in entries)}, adding: {len(content)}."
                )
            entries.append(content)
            _write_entries(self._paths[target], entries)
        finally:
            self._release(fh)
        return True, f"Added to {target}."

    def replace(self, target: str, old_text: str, new_content: str) -> Tuple[bool, str]:
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text or not new_content:
            return False, "old_text and new_content must not be empty."
        if target not in self._paths:
            return False, f"Unknown target '{target}'."

        fh = self._acquire(target)
        try:
            entries = _read_entries(self._paths[target])
            indices = [i for i, e in enumerate(entries) if old_text in e]
            if not indices:
                return False, "old_text not found in any entry."
            if len(indices) > 1:
                return False, "old_text matches multiple entries — be more specific."
            entries[indices[0]] = new_content
            _write_entries(self._paths[target], entries)
        finally:
            self._release(fh)
        return True, f"Replaced entry in {target}."

    def remove(self, target: str, old_text: str) -> Tuple[bool, str]:
        old_text = old_text.strip()
        if not old_text:
            return False, "old_text must not be empty."
        if target not in self._paths:
            return False, f"Unknown target '{target}'."

        fh = self._acquire(target)
        try:
            entries = _read_entries(self._paths[target])
            indices = [i for i, e in enumerate(entries) if old_text in e]
            if not indices:
                return False, "old_text not found."
            entries.pop(indices[0])
            _write_entries(self._paths[target], entries)
        finally:
            self._release(fh)
        return True, f"Removed entry from {target}."

    def read_all(self, target: str) -> List[str]:
        if target not in self._paths:
            return []
        return _read_entries(self._paths[target])

    def build_memory_context_block(self) -> str:
        """Build a <memory-context> fenced block for injection into the system prompt.

        Mirrors Hermess build_memory_context_block() in system_prompt.py:
        both MEMORY.md and USER.md entries are combined and wrapped in a
        <memory-context> ... </memory-context> XML fence that the model
        reads as persistent recall.

        Returns empty string if both stores are empty.
        """
        memory_entries = self.read_all("memory")
        user_entries = self.read_all("user")

        if not memory_entries and not user_entries:
            return ""

        sections: List[str] = []
        if memory_entries:
            sections.append("## Agent Memory")
            sections.extend(memory_entries)
        if user_entries:
            if sections:
                sections.append("")
            sections.append("## User Profile")
            sections.extend(user_entries)

        body = "\n".join(sections).strip()
        return f"<memory-context>\n{body}\n</memory-context>"

    def char_counts(self) -> dict:
        """Return current character usage for each store (useful for monitoring)."""
        return {
            target: sum(len(e) for e in self.read_all(target))
            for target in ("memory", "user")
        }
