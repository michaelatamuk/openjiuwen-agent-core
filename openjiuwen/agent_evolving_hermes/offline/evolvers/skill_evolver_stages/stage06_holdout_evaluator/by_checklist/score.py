from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List


@dataclass
class ChecklistScore:
    """Binary-checklist evaluation score.

    Dimensions
    ----------
    criteria_met   — count of concrete criteria that were satisfied.
    criteria_total — total count of criteria extracted from expected_behavior.
    failed_criteria — newline-separated list of failed criteria (for feedback).
    feedback        — actionable LLM feedback on failures.

    Composite
    ---------
    ``criteria_met / criteria_total`` — fraction of criteria passed.

    No correctness gate is applied.  The checklist judge scores compliance
    with explicit criteria; a separate holistic evaluation is the right
    place for a semantic correctness gate.
    """

    criteria_met: int = 0
    criteria_total: int = 1
    failed_criteria: str = ""
    feedback: str = ""

    DIM_NAMES: ClassVar[List[str]] = ["criteria_met", "criteria_total"]

    @property
    def composite(self) -> float:
        if self.criteria_total <= 0:
            return 0.5
        return min(1.0, max(0.0, self.criteria_met / self.criteria_total))

    def as_dict(self) -> Dict[str, float]:
        return {
            "criteria_met": float(self.criteria_met),
            "criteria_total": float(self.criteria_total),
            "pass_rate": self.composite,
        }
