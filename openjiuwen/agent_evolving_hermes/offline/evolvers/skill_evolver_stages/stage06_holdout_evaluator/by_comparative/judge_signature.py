import dspy


class ComparativeJudgeSignature(dspy.Signature):
    """Compare two agent responses to the same task against an expected-behavior rubric.

    Determine which response better satisfies the rubric.  Return a
    continuous preference score:

    * 0.0  — Response A (baseline) is clearly better.
    * 0.5  — Both responses are equally good (or both equally bad).
    * 1.0  — Response B (evolved) is clearly better.

    Values between 0 and 1 express degrees of preference.

    Evaluation criteria (in priority order)
    ----------------------------------------
    1. Factual correctness vs. the expected behavior.
    2. Coverage of required content / steps.
    3. Format and structure compliance.
    4. Conciseness (prefer the shorter response when quality is equal).

    Do NOT give extra credit for length.  Be specific in your reasoning
    about *what* makes one response better.
    """

    task_input: str = dspy.InputField(
        desc="The task the agent was given")
    expected_behavior: str = dspy.InputField(
        desc="Rubric describing what a good response looks like")
    baseline_output: str = dspy.InputField(
        desc="Response A — produced by the baseline (pre-evolution) skill")
    evolved_output: str = dspy.InputField(
        desc="Response B — produced by the evolved skill")

    preference: float = dspy.OutputField(
        desc="Score 0.0–1.0: 0.0 = Response A clearly better, "
             "0.5 = equal, 1.0 = Response B clearly better")
    reasoning: str = dspy.OutputField(
        desc="Specific, evidence-based explanation of the preference judgement. "
             "Cite concrete differences between the two responses.")
