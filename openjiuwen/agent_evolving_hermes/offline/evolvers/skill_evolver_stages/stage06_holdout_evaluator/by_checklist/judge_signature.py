import dspy


class ChecklistJudgeSignature(dspy.Signature):
    """Evaluate an agent's response by checking concrete binary criteria from the expected behavior.

    Step 1 — Extract 3–7 concrete, binary pass/fail criteria from
    ``expected_behavior``.  Criteria must be specific and verifiable, e.g.:
    "mentions error handling", "includes a numbered list of steps",
    "response is under 100 words", "states the time complexity".

    Step 2 — For each criterion, determine whether ``agent_output``
    satisfies it.

    Step 3 — Report the counts and list failed criteria.

    Focus on *specific, verifiable* requirements — not subjective quality
    judgements like "well written" or "comprehensive".
    """

    task_input: str = dspy.InputField(
        desc="The task the agent was given")
    expected_behavior: str = dspy.InputField(
        desc="Rubric describing what a good response looks like")
    agent_output: str = dspy.InputField(
        desc="The agent's actual response to be evaluated")

    criteria_met: int = dspy.OutputField(
        desc="Integer ≥ 0: number of concrete criteria that agent_output satisfies")
    criteria_total: int = dspy.OutputField(
        desc="Integer ≥ 1: total number of concrete criteria extracted from expected_behavior")
    failed_criteria: str = dspy.OutputField(
        desc="Newline-separated list of specific criteria that were NOT met. "
             "Empty string if all criteria passed.")
    feedback: str = dspy.OutputField(
        desc="Actionable feedback: which key criteria failed and how the response "
             "could be changed to satisfy them")
