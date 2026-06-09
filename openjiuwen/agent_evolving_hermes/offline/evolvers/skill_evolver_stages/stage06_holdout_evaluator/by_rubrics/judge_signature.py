from __future__ import annotations

import dspy


class JudgeSignature(dspy.Signature):
    """Score an agent response across five independent quality dimensions.

    Return five independent float scores (0.0–1.0) and brief feedback.
    Each dimension is scored independently — do not let one influence another.
    """

    task_input: str = dspy.InputField(desc="The task given to the agent")
    expected_behavior: str = dspy.InputField(desc="Rubric: what a good response looks like")
    agent_output: str = dspy.InputField(desc="The actual agent response to score")
    skill_text: str = dspy.InputField(desc="The skill instructions the agent was given")

    correctness: float = dspy.OutputField(desc="0.0–1.0: Did the agent do the right thing according to the task?")
    procedure_following: float = dspy.OutputField(
        desc="0.0–1.0: Did the agent follow the specified workflow in the skill?")
    format_adherence: float = dspy.OutputField(desc="0.0–1.0: Did the response match the expected output format and structure?")
    completeness: float = dspy.OutputField(desc="0.0–1.0: Did the response cover all required aspects of the task?")
    specificity: float = dspy.OutputField(desc="0.0–1.0: Are findings specific and actionable rather than vague?")
    feedback: str = dspy.OutputField(desc="One sentence identifying the main strength or most important weakness.")
