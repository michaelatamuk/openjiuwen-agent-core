from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional


class BaseAcceptanceGate(ABC):
    @abstractmethod
    def decide(self, improvement: float, evolved_score: float, skill_name: str, evolved_text: str,
               cross_run_delta: Optional[float], output_dir: Path, console):
        pass


    # ── Shared helper ─────────────────────────────────────────────────────────────
    @staticmethod
    def _trend_line(cross_run_delta: Optional[float]) -> Optional[str]:
        """Return a one-line trend note, or None if no prior run exists."""
        if cross_run_delta is None:
            return None

        if cross_run_delta >= 0:
            return (f"  Trend vs last run: {cross_run_delta:+.4f}"
                    f"  (candidate improving across runs)")

        return (f"  Trend vs last run: {cross_run_delta:+.4f}"
                f"  (candidate getting worse across runs)")
