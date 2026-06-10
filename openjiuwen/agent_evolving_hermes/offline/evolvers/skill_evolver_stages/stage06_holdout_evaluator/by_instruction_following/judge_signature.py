import dspy


class InstructionFollowingSignature(dspy.Signature):
    """Evaluate whether an agent's response followed the explicit instructions in the task.

    Identify every explicit, verifiable instruction in ``task_input``.
    Examples of explicit instructions:
    * "list exactly 3 items"
    * "respond in French"
    * "do not use bullet points"
    * "include a summary at the end"
    * "keep your response under 50 words"
    * "format the output as JSON"

    Do NOT evaluate content quality, factual accuracy, or completeness —
    ONLY whether explicit behavioural instructions were followed.

    If ``task_input`` contains no explicit instructions, set
    ``instructions_found = 0`` and ``instructions_followed = 0``.
    """

    task_input: str = dspy.InputField(
        desc="The task the agent was given (may contain explicit instructions)")
    agent_output: str = dspy.InputField(
        desc="The agent's actual response")

    instructions_found: int = dspy.OutputField(
        desc="Integer ≥ 0: number of explicit, verifiable instructions identified in task_input")
    instructions_followed: int = dspy.OutputField(
        desc="Integer ≥ 0, ≤ instructions_found: number of instructions correctly obeyed")
    violated_instructions: str = dspy.OutputField(
        desc="Newline-separated description of each violated instruction. "
             "Empty string if all instructions were followed or none were found.")
    feedback: str = dspy.OutputField(
        desc="Actionable feedback on how the response could be changed to "
             "comply with the violated instructions")
