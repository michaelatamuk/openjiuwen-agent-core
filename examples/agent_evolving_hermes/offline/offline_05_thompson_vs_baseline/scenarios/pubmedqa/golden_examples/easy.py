# coding: utf-8
# Easy — question can be answered from a single clear finding;
# correct verdict is strongly supported by the study summary provided.

GOLDEN_EXAMPLES_EASY = [
    {
        "task_input": (
            "Context: A randomised controlled trial enrolled 200 adults with "
            "type 2 diabetes. The metformin group showed a mean HbA1c reduction "
            "of 1.2% over 12 weeks; the placebo group showed 0.1% reduction "
            "(p < 0.001).\n"
            "Question: Does metformin reduce HbA1c in adults with type 2 diabetes?"
        ),
        "expected_behavior": (
            "yes\n"
            "The RCT demonstrated a statistically significant 1.2% HbA1c reduction "
            "with metformin versus 0.1% with placebo (p < 0.001)."
        ),
        "difficulty": "easy",
        "source": "pubmedqa-rct-clear-positive",
    },
    {
        "task_input": (
            "Context: Researchers tracked 500 non-smokers and 500 smokers for "
            "10 years. Lung cancer incidence was 0.4% in non-smokers versus "
            "8.7% in smokers (RR = 21.8, 95% CI 9.1–52.4).\n"
            "Question: Is cigarette smoking associated with increased lung cancer risk?"
        ),
        "expected_behavior": (
            "yes\n"
            "The cohort study found a 21.8-fold higher lung cancer incidence in "
            "smokers (8.7%) versus non-smokers (0.4%) over 10 years."
        ),
        "difficulty": "easy",
        "source": "pubmedqa-cohort-clear-positive",
    },
    {
        "task_input": (
            "Context: A double-blind crossover trial gave 60 healthy volunteers "
            "either vitamin C (1 g/day) or placebo for 8 weeks. Cold episode "
            "frequency was identical in both arms: 2.1 episodes per person "
            "(p = 0.94).\n"
            "Question: Does daily vitamin C supplementation prevent the common cold?"
        ),
        "expected_behavior": (
            "no\n"
            "The crossover trial found no difference in cold frequency between "
            "vitamin C (1 g/day) and placebo groups (2.1 vs 2.1 episodes, p = 0.94)."
        ),
        "difficulty": "easy",
        "source": "pubmedqa-rct-clear-negative",
    },
    {
        "task_input": (
            "Context: A meta-analysis of 12 RCTs (n = 3,400) evaluated aerobic "
            "exercise in adults with mild-to-moderate depression. Pooled effect "
            "size was d = 0.62 (95% CI 0.45–0.79) favouring exercise over control.\n"
            "Question: Is aerobic exercise effective for reducing depression symptoms?"
        ),
        "expected_behavior": (
            "yes\n"
            "A meta-analysis of 12 RCTs found a moderate-to-large pooled effect "
            "size (d = 0.62) favouring aerobic exercise over control for depression."
        ),
        "difficulty": "easy",
        "source": "pubmedqa-meta-analysis-positive",
    },
]
