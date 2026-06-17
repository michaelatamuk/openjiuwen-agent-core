# coding: utf-8
# Medium — answer requires weighing conflicting or partial evidence;
# correct verdict is "maybe" or a nuanced yes/no with a key caveat.

GOLDEN_EXAMPLES_MEDIUM = [
    {
        "task_input": (
            "Context: A prospective study of 800 adults over 5 years found that "
            "those who consumed ≥3 cups of coffee/day had a 22% lower risk of "
            "type 2 diabetes (HR 0.78, 95% CI 0.61–0.99). However, a concurrent "
            "RCT of 120 prediabetic patients showed no significant glycaemic "
            "improvement after 6 months of controlled coffee intake (p = 0.21).\n"
            "Question: Does coffee consumption reduce the risk of type 2 diabetes?"
        ),
        "expected_behavior": (
            "maybe\n"
            "Observational evidence shows a modest inverse association (HR 0.78), "
            "but the only RCT found no significant glycaemic benefit in prediabetic "
            "patients, leaving causality unresolved."
        ),
        "difficulty": "medium",
        "source": "pubmedqa-conflicting-evidence",
    },
    {
        "task_input": (
            "Context: A systematic review of 8 studies examined whether omega-3 "
            "supplements reduce cardiovascular mortality. Six studies reported "
            "reduced risk (RR range 0.71–0.89), one was null (RR 0.98), and one "
            "reported increased risk in patients on anticoagulants (RR 1.12). "
            "Heterogeneity was high (I² = 78%).\n"
            "Question: Do omega-3 supplements reduce cardiovascular mortality?"
        ),
        "expected_behavior": (
            "maybe\n"
            "Most studies suggest a protective effect, but high heterogeneity "
            "(I² = 78%) and one harmful result in anticoagulated patients prevent "
            "a definitive conclusion."
        ),
        "difficulty": "medium",
        "source": "pubmedqa-heterogeneous-meta-analysis",
    },
    {
        "task_input": (
            "Context: Researchers compared cognitive decline in 300 adults aged "
            "65–75 who used statins versus 300 matched non-users over 7 years. "
            "Statin users scored slightly better on the MMSE at year 7 "
            "(mean 26.1 vs 25.4, p = 0.048), but the difference disappeared after "
            "adjusting for cardiovascular comorbidities (p = 0.19).\n"
            "Question: Do statins protect against cognitive decline in older adults?"
        ),
        "expected_behavior": (
            "maybe\n"
            "An unadjusted benefit was observed (p = 0.048), but the effect became "
            "non-significant after adjusting for cardiovascular comorbidities, "
            "suggesting confounding rather than a direct protective effect."
        ),
        "difficulty": "medium",
        "source": "pubmedqa-confounding-adjusted",
    },
    {
        "task_input": (
            "Context: A phase II trial of 90 patients with treatment-resistant "
            "depression found that 60% responded to ketamine infusion at 48 hours "
            "versus 20% for placebo (p = 0.002). However, response durability was "
            "poor: only 30% maintained response at 4 weeks.\n"
            "Question: Is ketamine effective for treatment-resistant depression?"
        ),
        "expected_behavior": (
            "maybe\n"
            "Ketamine produced a strong short-term response (60% vs 20% at 48 h), "
            "but durability was limited — only 30% maintained response at 4 weeks, "
            "leaving long-term efficacy uncertain."
        ),
        "difficulty": "medium",
        "source": "pubmedqa-short-term-vs-durable",
    },
]
