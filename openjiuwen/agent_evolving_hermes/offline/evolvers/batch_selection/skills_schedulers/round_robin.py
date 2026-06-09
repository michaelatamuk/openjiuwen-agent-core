from __future__ import annotations

from typing import List, Tuple


# ── Round-robin (legacy) ──────────────────────────────────────────────────────

class RoundRobinSkillScheduler:
    """Return skills in registration order — preserves current behaviour.

    ``schedule()`` returns the skill list unchanged.
    ``record()`` is a no-op (outcomes are not used).
    """

    def __init__(self) -> None:
        self._names: List[str] = []

    def register(self, skill_names: List[str]) -> None:
        for name in skill_names:
            if name not in self._names:
                self._names.append(name)

    def schedule(self, skill_names: List[str]) -> List[str]:
        """Return *skill_names* in the order they were registered."""
        # Preserve registration order; honour any subset passed in
        registered_set = set(self._names)
        ordered = [n for n in self._names if n in set(skill_names)]
        extras = [n for n in skill_names if n not in registered_set]
        return ordered + extras

    def record(self, skill_name: str, improvement: float) -> None:
        pass  # round-robin ignores outcomes

    def rankings(self) -> List[Tuple[str, float]]:
        return [(name, 0.5) for name in self._names]

