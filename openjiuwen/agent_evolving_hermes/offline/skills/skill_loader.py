# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""DSPy SkillModule — wraps a SKILL.md as an optimisable DSPy module.

Mirrors hermes-agent-self-evolution evolution/skills/skill_module.py exactly.
"""
from pathlib import Path
from typing import Dict

import yaml


def load_skill(skill_path: Path) -> Dict[str, str]:
    """Read and parse a SKILL.md file.

    Returns dict with keys: raw, frontmatter_text, frontmatter, body, name, description.
    """
    raw = skill_path.read_text(encoding="utf-8", errors="replace")
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            fm_text = raw[3:end]
            body = raw[end + 4:].lstrip("\n")
            try:
                fm = yaml.safe_load(fm_text) or {}
            except yaml.YAMLError:
                fm = {}
            return {
                "raw": raw,
                "frontmatter_text": fm_text,
                "frontmatter": fm,
                "body": body,
                "name": fm.get("name", skill_path.parent.name),
                "description": fm.get("description", ""),
            }
    return {
        "raw": raw,
        "frontmatter_text": "",
        "frontmatter": {},
        "body": raw,
        "name": skill_path.parent.name,
        "description": "",
    }
