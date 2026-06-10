from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, List


@dataclass
class InstructionFollowingScore:
    """Instruction-following compliance score.

    Dimensions
    ----------
    instructions_found    — number of explicit instructions identified in task_input.
    instructions_followed — number of those instructions that were obeyed.
    violated_instructions — newline-separated description of violations.
    feedback              — actionable feedback from the LLM judge.

    Composite
    ---------
    ``instructions_followed / instructions_found``.
    Returns 1.0 when no explicit instructions are found (no constraints → full compliance).

    Why separate from holistic
    --------------------------
    LLM holistic judges conflate content quality with instruction compliance.
    A factually excellent response that violates an explicit format constraint
    (e.g. "list exactly 3 items") should be penalised for the violation
    independently of its content quality.  This judge isolates that signal.
    """

    instructions_found: int = 0
    instructions_followed: int = 0
    violated_instructions: str = ""
    feedback: str = ""

    DIM_NAMES: ClassVar[List[str]] = ["instructions_found", "instructions_followed"]

    @property
    def composite(self) -> float:
        if self.instructions_found <= 0:
            return 1.0  # No instructions given → full compliance by default
        return min(1.0, max(0.0, self.instructions_followed / self.instructions_found))

    def as_dict(self) -> Dict[str, float]:
        return {
            "instructions_found": float(self.instructions_found),
            "instructions_followed": float(self.instructions_followed),
            "compliance_rate": self.composite,
        }
