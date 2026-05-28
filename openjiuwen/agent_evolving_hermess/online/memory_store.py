# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Simple flat-file memory store.

Mirrors Hermess MemoryStore exactly.
Two stores: "memory" (agent observations) and "user" (user profile).
Uses fcntl file locking and atomic writes.

New vs original implementation:
  - Frozen snapshot pattern (load_from_disk / get_snapshot_block):
      System prompt uses a snapshot taken at session start that never changes
      mid-session, ensuring byte-identical prefix cache hits across all turns.
      Mirrors Hermess memory_tool.py _system_prompt_snapshot design.
  - Drift detection:
      detect_drift() notices when a memory file is modified externally
      between load and the current call, creating a .bak.<timestamp> backup.
      Mirrors Hermess memory_tool.py external-drift detection (lines 516-569).
"""
from __future__ import annotations

import fcntl
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Dict, List, Optional, Tuple

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
    # Load-time deduplication, preserving order (mirrors Hermess line 158)
    seen: Dict[str, None] = {}
    for e in raw.split(_ENTRY_DELIMITER):
        e = e.strip()
        if e:
            seen[e] = None
    return list(seen.keys())


def _write_entries(path: Path, entries: List[str]) -> None:
    _atomic_write(path, _ENTRY_DELIMITER.join(entries))


class MemoryStore:
    """File-backed key-value memory. Mirrors Hermess MemoryStore.

    Frozen snapshot:
        Call ``load_from_disk()`` once at session start.  The snapshot is
        stored in ``_system_prompt_snapshot`` and never updated mid-session,
        ensuring prefix-cache stability.  Subsequent ``add/replace/remove``
        calls mutate the live disk state but NOT the snapshot.

        Use ``get_snapshot_block()`` to inject the frozen state into the
        system prompt, and ``build_memory_context_block()`` to get the
        *current* live state (e.g. for tool call responses).

    Drift detection:
        Call ``detect_drift(target)`` to check whether a memory file was
        modified externally since it was loaded.  If drift is detected, a
        ``.bak.<timestamp>`` backup is created and the method returns True.
    """

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
        # Frozen snapshot: taken at load_from_disk(), never updated mid-session
        self._system_prompt_snapshot: Optional[str] = None
        # File mtimes at snapshot time, used for drift detection
        self._snapshot_mtimes: Dict[str, Optional[float]] = {"memory": None, "user": None}

    # ── Locking ───────────────────────────────────────────────────────────────

    def _acquire(self, target: str) -> IO:
        lock_path = self._lock_paths[target]
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "w")
        fcntl.flock(fh, fcntl.LOCK_EX)
        return fh

    def _release(self, fh: IO) -> None:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()

    # ── Frozen snapshot ───────────────────────────────────────────────────────

    def load_from_disk(self) -> None:
        """Load memory from disk and freeze a system-prompt snapshot.

        Call once at session start (e.g. in gateway hydration).
        The snapshot is byte-identical across the entire session so the
        prefix cache key never changes.

        Subsequent ``add/replace/remove`` writes mutate disk but not the snapshot.
        """
        self._snapshot_mtimes = {
            t: (self._paths[t].stat().st_mtime if self._paths[t].exists() else None)
            for t in ("memory", "user")
        }
        self._system_prompt_snapshot = self.build_memory_context_block()

    def get_snapshot_block(self) -> str:
        """Return the frozen system-prompt snapshot (from ``load_from_disk()``).

        Returns the *current live* block if load_from_disk() was never called,
        so callers don't need a try/except.
        """
        if self._system_prompt_snapshot is not None:
            return self._system_prompt_snapshot
        return self.build_memory_context_block()

    # ── Drift detection ───────────────────────────────────────────────────────

    def detect_drift(self, target: str) -> bool:
        """Check whether ``target`` file was modified externally since load.

        If drift is detected:
          1. Creates a ``.bak.<timestamp>`` backup of the current file.
          2. Returns True.

        Returns False if no drift, or if load_from_disk() was never called.
        """
        if self._snapshot_mtimes.get(target) is None:
            return False
        path = self._paths.get(target)
        if not path or not path.exists():
            return False
        current_mtime = path.stat().st_mtime
        if current_mtime <= self._snapshot_mtimes[target]:  # type: ignore[operator]
            return False
        # Drift detected — create backup
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = path.parent / f"{path.name}.bak.{ts}"
        try:
            import shutil
            shutil.copy2(str(path), str(backup))
        except OSError:
            pass
        # Update mtime record so we don't keep re-backing-up
        self._snapshot_mtimes[target] = current_mtime
        return True

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add(self, target: str, content: str) -> Tuple[bool, str]:
        content = content.strip()
        if not content:
            return False, "Content must not be empty."
        if target not in self._paths:
            return False, f"Unknown target '{target}'. Use 'memory' or 'user'."

        fh = self._acquire(target)
        try:
            entries = _read_entries(self._paths[target])
            # Exact duplicate rejection (mirrors Hermess memory_tool.py lines 321-323)
            if content in entries:
                return False, "Entry already exists (no duplicate added)."
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

    # ── Reads ─────────────────────────────────────────────────────────────────

    def read_all(self, target: str) -> List[str]:
        if target not in self._paths:
            return []
        return _read_entries(self._paths[target])

    def build_memory_context_block(self) -> str:
        """Build a <memory-context> fenced block for injection into the system prompt.

        Mirrors Hermess build_memory_context_block() in system_prompt.py:
        both MEMORY.md and USER.md entries are combined and wrapped in a
        <memory-context> ... </memory-context> XML fence.

        This reflects LIVE disk state.  Use ``get_snapshot_block()`` for the
        session-start frozen snapshot needed for prefix-cache stability.

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
