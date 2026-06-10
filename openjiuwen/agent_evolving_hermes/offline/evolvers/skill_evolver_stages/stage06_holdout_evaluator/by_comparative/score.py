from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List


@dataclass
class ComparativeScore:
    """Preference score from a head-to-head comparison judge.

    Dimensions
    ----------
    preference — 0.0 = baseline clearly better, 0.5 = equal/tie,
                 1.0 = evolved clearly better.
    reasoning  — LLM explanation for the preference judgement.

    Composite
    ---------
    The ``composite`` property is identical to ``preference``.

    Interpretation
    --------------
    A ``preference`` of 0.5 means the evolved skill is no better or worse
    than the baseline on this example.  The baseline for all comparative
    runs is therefore 0.5.  An improvement of +0.1 means the evolved skill
    wins on ~60 % of holdout examples.
    """

    preference: float = 0.5
    reasoning: str = ""

    DIM_NAMES: ClassVar[List[str]] = ["preference"]

    @property
    def composite(self) -> float:
        return min(1.0, max(0.0, self.preference))

    def as_dict(self) -> Dict[str, float]:
        return {"preference": self.preference}
