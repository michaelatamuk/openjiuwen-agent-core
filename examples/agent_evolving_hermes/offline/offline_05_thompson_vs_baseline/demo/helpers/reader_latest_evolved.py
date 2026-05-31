from pathlib import Path
from typing import Optional


def _read_latest_evolved(output_dir: Path, skill_name: str) -> Optional[str]:
    for pattern in ("*/evolved_skill.md", "*/evolved_REGRESSION.md", "*/evolved_FAILED.md"):
        candidates = sorted((output_dir / skill_name).glob(pattern))
        if candidates:
            return candidates[-1].read_text()
    return None
