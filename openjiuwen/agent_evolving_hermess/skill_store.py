# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""SKILL.md filesystem operations.

Mirrors Hermess skill_manager_tool.py internal helpers.
Handles concurrent access with per-skill asyncio locks and atomic writes.

New vs original implementation:
  - UsageSidecar dataclass (.usage.json per skill) with full telemetry
  - Lifecycle states: SKILL_STATE_ACTIVE / STALE / ARCHIVED
  - skill_delete() with absorbed_into consolidation-intent tracking
  - skill_archive() / skill_restore() for reversible deactivation
  - skill_get_usage() / skill_set_pinned() helpers
  - skill_create() records write origin from provenance ContextVar
  - skill_list() filters archived skills by default
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ── Constants ─────────────────────────────────────────────────────────────────

_SKILL_LOCKS: Dict[str, asyncio.Lock] = {}
_ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MAX_SKILL_SIZE = 100_000
_ARCHIVE_SUBDIR = ".archive"

SKILL_STATE_ACTIVE = "active"
SKILL_STATE_STALE = "stale"
SKILL_STATE_ARCHIVED = "archived"


# ── Usage sidecar ─────────────────────────────────────────────────────────────


@dataclass
class UsageSidecar:
    """Per-skill telemetry sidecar, mirroring Hermess skill_usage.py.

    Written to ``<skill_dir>/.usage.json``.

    Lifecycle states mirror Hermess:
      active   — in regular use
      stale    — unused for stale_after_days (default 30)
      archived — unused for archive_after_days (default 90); moved to .archive/
    """

    created_by: str = "user"           # "user" | "agent"
    use_count: int = 0
    view_count: int = 0
    patch_count: int = 0
    pinned: bool = False               # pin blocks deletion/archive
    state: str = SKILL_STATE_ACTIVE
    archived_at: Optional[str] = None  # ISO-8601 timestamp when archived
    absorbed_into: Optional[str] = None  # If deleted via consolidation, target skill


def _read_usage(skill_dir: Path) -> UsageSidecar:
    p = skill_dir / ".usage.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            known = {f for f in UsageSidecar.__dataclass_fields__}
            return UsageSidecar(**{k: v for k, v in data.items() if k in known})
        except Exception:
            pass
    return UsageSidecar()


def _write_usage(skill_dir: Path, usage: UsageSidecar) -> None:
    _atomic_write(skill_dir / ".usage.json", json.dumps(asdict(usage), indent=2))


# ── Internal helpers ─────────────────────────────────────────────────────────


def _get_lock(skill_name: str) -> asyncio.Lock:
    if skill_name not in _SKILL_LOCKS:
        _SKILL_LOCKS[skill_name] = asyncio.Lock()
    return _SKILL_LOCKS[skill_name]


def _default_skills_root() -> Path:
    return Path.home() / ".jiuwen" / "skills"


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


def _parse_frontmatter(content: str) -> Tuple[Optional[dict], str]:
    """Split YAML frontmatter from markdown body.

    Returns (frontmatter_dict, body) or (None, content) if no frontmatter.
    """
    if not content.startswith("---"):
        return None, content
    end = content.find("\n---", 3)
    if end == -1:
        return None, content
    try:
        fm = yaml.safe_load(content[3:end])
        return fm or {}, content[end + 4:].lstrip("\n")
    except yaml.YAMLError:
        return None, content


def _validate_frontmatter(content: str) -> Optional[str]:
    """Return error string or None if frontmatter is valid."""
    fm, _ = _parse_frontmatter(content)
    if fm is None:
        return "SKILL.md must begin with YAML frontmatter (--- ... ---)."
    if not fm.get("name"):
        return "Frontmatter must contain a 'name' field."
    if not fm.get("description"):
        return "Frontmatter must contain a 'description' field."
    if len(fm.get("description", "")) > 1024:
        return "description must be ≤ 1024 characters."
    return None


def _is_immutable(content: str) -> bool:
    fm, _ = _parse_frontmatter(content)
    return bool(fm and fm.get("immutable"))


def _find_skill(name: str, skills_root: Path) -> Optional[Path]:
    """Return the skill directory path if it exists (not archived)."""
    direct = skills_root / name
    if (direct / "SKILL.md").exists():
        return direct
    # Search one level of category subdirectories (skip .archive)
    for category_dir in skills_root.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        candidate = category_dir / name
        if (candidate / "SKILL.md").exists():
            return candidate
    return None


def _find_archived(name: str, skills_root: Path) -> Optional[Path]:
    """Return the archived skill directory path, or None."""
    archive_root = skills_root / _ARCHIVE_SUBDIR
    direct = archive_root / name
    if (direct / "SKILL.md").exists():
        return direct
    for category_dir in archive_root.iterdir() if archive_root.exists() else []:
        if not category_dir.is_dir():
            continue
        candidate = category_dir / name
        if (candidate / "SKILL.md").exists():
            return candidate
    return None


# ── Public API ────────────────────────────────────────────────────────────────


async def skill_read(name: str, skills_root: Path) -> Optional[str]:
    """Read and return raw SKILL.md content, or None if not found."""
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return None
    usage = _read_usage(skill_dir)
    usage.view_count += 1
    _write_usage(skill_dir, usage)
    return (skill_dir / "SKILL.md").read_text(encoding="utf-8")


async def skill_create(
    name: str,
    content: str,
    skills_root: Path,
    category: Optional[str] = None,
    protected_names: List[str] = (),
) -> Tuple[bool, str]:
    """Create a new skill. Returns (success, message).

    Records write origin from the provenance ContextVar:
      - foreground        → created_by='user'    (not curator-managed)
      - background_review → created_by='agent'   (curator-eligible)
    """
    if not _NAME_RE.match(name):
        return False, f"Invalid skill name '{name}'."
    err = _validate_frontmatter(content)
    if err:
        return False, err
    if len(content) > _MAX_SKILL_SIZE:
        return False, f"Content exceeds {_MAX_SKILL_SIZE} char limit."
    if _find_skill(name, skills_root):
        return False, f"Skill '{name}' already exists."

    # Import here to avoid circular imports at module level
    from openjiuwen.agent_evolving_hermess.provenance import get_write_origin

    origin = get_write_origin()
    created_by = "agent" if origin == "background_review" else "user"

    async with _get_lock(name):
        skill_dir = (
            (skills_root / category / name) if category else (skills_root / name)
        )
        _atomic_write(skill_dir / "SKILL.md", content)
        usage = UsageSidecar(created_by=created_by)
        _write_usage(skill_dir, usage)

    return True, f"Created skill '{name}' at {skill_dir}."


async def skill_edit(
    name: str,
    content: str,
    skills_root: Path,
    protected_names: List[str] = (),
) -> Tuple[bool, str]:
    """Full replacement of SKILL.md. Returns (success, message)."""
    if name in protected_names:
        return (
            False,
            f"Skill '{name}' is protected and cannot be edited by background review.",
        )
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return False, f"Skill '{name}' not found."
    existing = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    if _is_immutable(existing):
        return False, f"Skill '{name}' is immutable (bundled or hub-installed)."
    err = _validate_frontmatter(content)
    if err:
        return False, err
    if len(content) > _MAX_SKILL_SIZE:
        return False, f"Content exceeds {_MAX_SKILL_SIZE} char limit."
    if len(content) > len(existing) * 1.20:
        return False, "Edit rejected: would grow skill by > 20%."

    async with _get_lock(name):
        _atomic_write(skill_dir / "SKILL.md", content)
        usage = _read_usage(skill_dir)
        usage.patch_count += 1
        _write_usage(skill_dir, usage)
    return True, f"Edited skill '{name}'."


async def skill_patch(
    name: str,
    old_string: str,
    new_string: str,
    skills_root: Path,
    replace_all: bool = False,
    protected_names: List[str] = (),
) -> Tuple[bool, str]:
    """String-replace patch of SKILL.md. Returns (success, message)."""
    if name in protected_names:
        return False, f"Skill '{name}' is protected."
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return False, f"Skill '{name}' not found."
    path = skill_dir / "SKILL.md"
    content = path.read_text(encoding="utf-8")
    if _is_immutable(content):
        return False, f"Skill '{name}' is immutable."
    if old_string not in content:
        return False, f"old_string not found in skill '{name}'."

    if replace_all:
        new_content = content.replace(old_string, new_string)
    else:
        new_content = content.replace(old_string, new_string, 1)

    err = _validate_frontmatter(new_content)
    if err:
        return False, err
    if len(new_content) > _MAX_SKILL_SIZE:
        return False, "Patched content exceeds size limit."
    if len(new_content) > len(content) * 1.20:
        return False, "Patch rejected: would grow skill by > 20%."

    async with _get_lock(name):
        _atomic_write(path, new_content)
        usage = _read_usage(skill_dir)
        usage.patch_count += 1
        _write_usage(skill_dir, usage)
    return True, f"Patched skill '{name}'."


async def skill_delete(
    name: str,
    skills_root: Path,
    protected_names: List[str] = (),
    absorbed_into: str = "",
) -> Tuple[bool, str]:
    """Permanently delete a skill directory.

    ``absorbed_into`` mirrors Hermess skill_manager_tool.py absorbed_into
    parameter: declare which skill absorbed this skill's content (if any).
    This intent is recorded in .usage.json before deletion so audit logs
    can reconstruct consolidation history.

    Pinned skills cannot be deleted — use skill_set_pinned() first.

    Returns (success, message).
    """
    if name in protected_names:
        return False, f"Skill '{name}' is protected."
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return False, f"Skill '{name}' not found."
    existing = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    if _is_immutable(existing):
        return False, f"Skill '{name}' is immutable and cannot be deleted."
    usage = _read_usage(skill_dir)
    if usage.pinned:
        return False, (
            f"Skill '{name}' is pinned — unpin it with skill_set_pinned() before deleting."
        )

    async with _get_lock(name):
        if absorbed_into:
            usage.absorbed_into = absorbed_into
            _write_usage(skill_dir, usage)
        shutil.rmtree(skill_dir)
    return True, (
        f"Deleted skill '{name}'"
        + (f" (content absorbed into '{absorbed_into}')" if absorbed_into else "")
        + "."
    )


async def skill_archive(
    name: str,
    skills_root: Path,
    protected_names: List[str] = (),
) -> Tuple[bool, str]:
    """Move a skill to the .archive/ subdirectory (reversible).

    Pinned skills cannot be archived.
    Archived skills are hidden from skill_list() by default.

    Returns (success, message).
    """
    if name in protected_names:
        return False, f"Skill '{name}' is protected."
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return False, f"Skill '{name}' not found."
    existing = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    if _is_immutable(existing):
        return False, f"Skill '{name}' is immutable and cannot be archived."
    usage = _read_usage(skill_dir)
    if usage.pinned:
        return False, f"Skill '{name}' is pinned — unpin before archiving."
    if usage.state == SKILL_STATE_ARCHIVED:
        return False, f"Skill '{name}' is already archived."

    archive_dest = skills_root / _ARCHIVE_SUBDIR / name
    if archive_dest.exists():
        return False, f"Archive collision: {archive_dest} already exists."

    async with _get_lock(name):
        usage.state = SKILL_STATE_ARCHIVED
        usage.archived_at = datetime.now(timezone.utc).isoformat()
        _write_usage(skill_dir, usage)
        archive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(skill_dir), str(archive_dest))
    return True, f"Archived skill '{name}' to {archive_dest}."


async def skill_restore(
    name: str,
    skills_root: Path,
) -> Tuple[bool, str]:
    """Restore an archived skill back to active.

    Returns (success, message).
    """
    archived_dir = _find_archived(name, skills_root)
    if not archived_dir:
        return False, f"No archived skill '{name}' found."
    if _find_skill(name, skills_root):
        return False, f"Skill '{name}' already exists as an active skill — cannot restore."

    restore_dest = skills_root / name

    async with _get_lock(name):
        restore_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(archived_dir), str(restore_dest))
        usage = _read_usage(restore_dest)
        usage.state = SKILL_STATE_ACTIVE
        usage.archived_at = None
        _write_usage(restore_dest, usage)
    return True, f"Restored skill '{name}' from archive."


async def skill_get_usage(
    name: str,
    skills_root: Path,
) -> Optional[UsageSidecar]:
    """Return the UsageSidecar for a skill, or None if not found."""
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return None
    return _read_usage(skill_dir)


async def skill_set_pinned(
    name: str,
    skills_root: Path,
    pinned: bool,
) -> Tuple[bool, str]:
    """Pin or unpin a skill.

    Pinned skills cannot be deleted or archived.
    Returns (success, message).
    """
    skill_dir = _find_skill(name, skills_root)
    if not skill_dir:
        return False, f"Skill '{name}' not found."
    async with _get_lock(name):
        usage = _read_usage(skill_dir)
        usage.pinned = pinned
        _write_usage(skill_dir, usage)
    verb = "Pinned" if pinned else "Unpinned"
    return True, f"{verb} skill '{name}'."


async def skill_list(
    skills_root: Path,
    include_archived: bool = False,
) -> List[str]:
    """Return sorted list of all skill names.

    By default, archived skills (.archive/) are excluded.
    Pass include_archived=True to include them.
    """
    names = []
    if not skills_root.exists():
        return names
    for item in skills_root.rglob("SKILL.md"):
        # Skip .archive/ subtree unless explicitly requested
        parts = item.parts
        if _ARCHIVE_SUBDIR in parts and not include_archived:
            continue
        names.append(item.parent.name)
    return sorted(names)


async def build_skills_system_prompt(skills_root: Path) -> str:
    """Build a compact skills index for injection into the agent system prompt.

    Mirrors Hermess prompt_builder.py build_skills_system_prompt():
    - Iterates all SKILL.md files
    - Extracts name + description from frontmatter
    - Marks bundled/hub-installed skills with [bundled]
    - Returns a markdown-formatted index string

    Returns empty string if no skills exist.

    The returned block is suitable for the STABLE tier of the system prompt
    (it changes only when skills are added/removed/renamed).
    """
    names = await skill_list(skills_root)
    if not names:
        return ""

    lines: List[str] = ["## Available Skills", ""]
    for name in names:
        content = await skill_read(name, skills_root)
        if not content:
            continue
        fm, _ = _parse_frontmatter(content)
        desc = ""
        immutable = False
        if fm and isinstance(fm, dict):
            desc = str(fm.get("description", ""))
            immutable = bool(fm.get("immutable", False))
        tag = " [bundled]" if immutable else ""
        if desc:
            lines.append(f"- **{name}**{tag}: {desc}")
        else:
            lines.append(f"- **{name}**{tag}")

    lines.append("")
    lines.append(
        "Use the skill_view tool to read a skill's full instructions before executing a task."
    )
    return "\n".join(lines)
