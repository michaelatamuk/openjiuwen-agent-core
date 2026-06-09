import dspy


class JudgeSignature(dspy.Signature):
    """Evaluate an agent's response against an expected behavior rubric.

    Score the response on three dimensions (0.0 to 1.0 each):
    1. correctness: Did the response correctly address the task?
    2. procedure_following: Did it follow the expected approach/procedure?
    3. conciseness: Was it appropriately concise without omitting important info?

    Also provide specific, actionable feedback on what could be improved.
    """

    task_input: str = dspy.InputField(desc="The task the agent was given")
    expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
    agent_output: str = dspy.InputField(desc="The agent's actual response")
    skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")

    correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
    procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
    conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
    feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")
