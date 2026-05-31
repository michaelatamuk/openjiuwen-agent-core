from pathlib import Path
from typing import Optional


def _read_latest_evolved(output_dir: Path, skill_name: str) -> Optional[str]:
    candidates = sorted((output_dir / skill_name).glob("*/evolved_skill.md"))
    if not candidates:
        # Accept regression too, to always show something
        candidates = sorted((output_dir / skill_name).glob("*/evolved_REGRESSION.md"))
    return candidates[-1].read_text() if candidates else None
