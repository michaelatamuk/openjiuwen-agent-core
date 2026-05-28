# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""SKILL.md filesystem operations.

Mirrors Hermess skill_manager_tool.py internal helpers.
Handles concurrent access with per-skill asyncio locks and atomic writes.
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_SKILL_LOCKS: Dict[str, asyncio.Lock] = {}
_ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MAX_SKILL_SIZE = 100_000


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
    """Return the skill directory path if it exists."""
    direct = skills_root / name
    if (direct / "SKILL.md").exists():
        return direct
    # Search one level of category subdirectories
    for category_dir in skills_root.iterdir():
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
    return (skill_dir / "SKILL.md").read_text(encoding="utf-8")


async def skill_create(
    name: str,
    content: str,
    skills_root: Path,
    category: Optional[str] = None,
    protected_names: List[str] = (),
) -> Tuple[bool, str]:
    """Create a new skill. Returns (success, message)."""
    if not _NAME_RE.match(name):
        return False, f"Invalid skill name '{name}'."
    err = _validate_frontmatter(content)
    if err:
        return False, err
    if len(content) > _MAX_SKILL_SIZE:
        return False, f"Content exceeds {_MAX_SKILL_SIZE} char limit."
    if _find_skill(name, skills_root):
        return False, f"Skill '{name}' already exists."

    async with _get_lock(name):
        skill_dir = (
            (skills_root / category / name) if category else (skills_root / name)
        )
        _atomic_write(skill_dir / "SKILL.md", content)
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
    return True, f"Patched skill '{name}'."


async def skill_list(skills_root: Path) -> List[str]:
    """Return sorted list of all skill names."""
    names = []
    if not skills_root.exists():
        return names
    for item in skills_root.rglob("SKILL.md"):
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
