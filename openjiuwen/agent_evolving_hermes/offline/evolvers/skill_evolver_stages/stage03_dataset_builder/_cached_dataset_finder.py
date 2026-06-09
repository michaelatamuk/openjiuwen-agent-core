from __future__ import annotations

from pathlib import Path
from typing import Optional


def _find_cached_dataset(skill_name: str, output_dir: Path) -> Optional[Path]:
    """Return the most recent cached dataset directory for this skill, or None."""
    skill_base = output_dir / skill_name
    if not skill_base.exists():
        return None

    candidates = sorted((d / "dataset" for d in skill_base.iterdir() if d.is_dir()),
                        key=lambda p: p.parent.name,
                        reverse=True)

    for candidate in candidates:
        if (candidate / "train.jsonl").exists():
            return candidate

    return None