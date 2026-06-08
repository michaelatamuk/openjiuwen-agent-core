from typing import Optional


class BaseAcceptanceGate:
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
