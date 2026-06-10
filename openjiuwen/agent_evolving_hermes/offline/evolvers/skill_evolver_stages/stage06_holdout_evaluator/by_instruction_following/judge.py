from __future__ import annotations

import dspy

from .judge_signature import InstructionFollowingSignature
from .score import InstructionFollowingScore


class InstructionFollowingLLMJudge:
    """Instruction-following LLM judge.

    Prompts the LLM to identify explicit instructions in ``task_input``
    and check which were obeyed in ``agent_output``.  Content quality
    and factual correctness are explicitly out of scope — only behavioural
    instruction compliance is assessed.

    Usage
    -----
    ::

        judge = InstructionFollowingLLMJudge(model="deepseek/deepseek-chat")
        ifs = judge.score(
            task_input=ex.task_input,
            agent_output=pred.output,
        )
        print(ifs.composite, ifs.violated_instructions)
    """

    def __init__(self, model: str):
        self._judge = dspy.ChainOfThought(InstructionFollowingSignature)
        self._model = model

    def score(self,
              task_input: str,
              agent_output: str) -> InstructionFollowingScore:
        lm = dspy.LM(self._model)
        with dspy.context(lm=lm):
            result = self._judge(
                task_input=task_input,
                agent_output=agent_output,
            )
        instructions_found = max(0, int(getattr(result, "instructions_found", 0)))
        instructions_followed = max(
            0, min(instructions_found, int(getattr(result, "instructions_followed", 0)))
        )
        return InstructionFollowingScore(
            instructions_found=instructions_found,
            instructions_followed=instructions_followed,
            violated_instructions=getattr(result, "violated_instructions", ""),
            feedback=getattr(result, "feedback", ""),
        )
