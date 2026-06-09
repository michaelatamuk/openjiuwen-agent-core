import dspy


class JudgeSignature(dspy.Signature):
    """Score an agent response against the expected behavior rubric.

    Return three independent float scores (0.0–1.0) and brief feedback.
    """

    task_input: str = dspy.InputField(desc="The task given to the agent")
    expected_behavior: str = dspy.InputField(desc="Rubric: what a good response looks like")
    agent_output: str = dspy.InputField(desc="The actual agent response to score")
    skill_text: str = dspy.InputField(desc="The skill instructions the agent was given")

    correctness: float = dspy.OutputField(desc="0.0–1.0: Did the agent do the right thing?")
    procedure_following: float = dspy.OutputField(desc="0.0–1.0: Did it follow the specified workflow?")
    conciseness: float = dspy.OutputField(desc="0.0–1.0: Was the response appropriately concise?")
    feedback: str = dspy.OutputField(desc="One sentence explaining the main strength or weakness.")
