# coding: utf-8
# Easy examples — issues any careful reader would notice (baseline gets partial credit)
GOLDEN_EXAMPLES_EASY = [
    {
        "task_input": (
            "Review this excerpt from a paper's Results section:\n\n"
            "\"The treatment group showed improvement (p < 0.05). "
            "The control group did not improve. Therefore the treatment works.\"\n\n"
            "No specific statistical test is named. No effect size is reported. "
            "Sample size: n = 8 per group."
        ),
        "expected_behavior": (
            "Two issues: (1) The statistical test is unnamed — state which test was used "
            "(t-test, Mann-Whitney U, etc.) and report the exact statistic and p-value "
            "(e.g. t(14)=2.31, p=0.037). (2) `effect_size` is missing — a significant "
            "p-value with n=8 per group is uninformative without `Cohen's_d` or similar. "
            "The conclusion 'the treatment works' is too strong for a pilot study of 8; "
            "write 'preliminary evidence suggests'."
        ),
        "difficulty": "easy",
        "source": "paper-review-stats-basics",
    },
    {
        "task_input": (
            "Review this abstract:\n\n"
            "\"We investigated whether mindfulness training improves academic performance. "
            "Students in the mindfulness group had higher GPA at the end of term. "
            "These results demonstrate that mindfulness causes better academic outcomes "
            "and should be mandatory in schools.\"\n\n"
            "Study design: students self-selected into the mindfulness group."
        ),
        "expected_behavior": (
            "Two critical errors: (1) Self-selection makes this an observational study, "
            "not an experiment — students who chose mindfulness likely differ in motivation "
            "and prior GPA (`confounding_variables`, `selection_bias`). Causal language "
            "'causes' is unjustified; replace with 'is associated with'. "
            "(2) Policy recommendation ('should be mandatory') goes beyond what the data "
            "support — that exceeds the scope of a single observational study."
        ),
        "difficulty": "easy",
        "source": "paper-review-causal-language",
    },
    {
        "task_input": (
            "Review this methods section:\n\n"
            "\"We used a questionnaire to measure employee satisfaction. "
            "The questionnaire was developed by the research team for this study. "
            "Participants rated 12 items on a 5-point Likert scale. "
            "Data were analysed using SPSS.\"\n\n"
            "No reliability or validity information is provided for the questionnaire."
        ),
        "expected_behavior": (
            "New questionnaire with no validation evidence: report `Cronbach_alpha` for "
            "internal consistency (acceptable threshold α ≥ 0.70), test-retest reliability "
            "if the construct is stable, and `construct_validity` (e.g. convergent validity "
            "with an established scale). Without these, readers cannot judge whether the "
            "instrument measures what it claims to measure. Mention of 'SPSS' is not a "
            "substitute for reporting the specific analyses run."
        ),
        "difficulty": "easy",
        "source": "paper-review-instrument-validation",
    },
    {
        "task_input": (
            "Review this paper's figure description:\n\n"
            "\"Figure 3 shows the results across all conditions. "
            "Error bars are shown.\"\n\n"
            "The figure shows bar charts with error bars but the caption does not "
            "specify what the error bars represent, nor is it stated in the methods."
        ),
        "expected_behavior": (
            "Error bars are ambiguous without definition — they could represent "
            "`standard_deviation`, `standard_error_of_mean`, or `95%_confidence_interval`, "
            "which have very different interpretations for reader inference. "
            "APA and most style guides require the figure caption or methods to explicitly "
            "state 'Error bars represent ±1 SD' (or SE, or 95% CI). "
            "Standard error bars are often misread as confidence intervals, inflating "
            "the apparent precision of the estimates."
        ),
        "difficulty": "easy",
        "source": "paper-review-figure-caption",
    },
]
