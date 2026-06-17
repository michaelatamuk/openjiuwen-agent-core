# coding: utf-8
# Hard — question requires synthesising subtle mechanistic or epidemiological
# nuance; expected_behavior distinguishes a specific sub-population qualifier
# or mechanistic caveat that the baseline will omit or hedge excessively.

GOLDEN_EXAMPLES_HARD = [
    {
        "task_input": (
            "Context: A genome-wide association study (GWAS) of 15,000 participants "
            "identified 3 SNPs in the CYP2C19 gene strongly associated with "
            "clopidogrel non-response (OR 3.4, p = 2 × 10⁻¹²). A subsequent RCT "
            "of 2,200 ACS patients showed that CYP2C19 loss-of-function carriers "
            "randomised to clopidogrel had a 34% higher MACE rate versus ticagrelor "
            "(p = 0.003), while non-carriers showed no significant difference "
            "(p = 0.42).\n"
            "Question: Should CYP2C19 genotype guide antiplatelet therapy selection "
            "in acute coronary syndrome patients?"
        ),
        "expected_behavior": (
            "yes\n"
            "In CYP2C19 loss-of-function carriers, clopidogrel was associated with "
            "34% higher MACE versus ticagrelor (p = 0.003), while non-carriers "
            "showed no difference, supporting genotype-guided selection."
        ),
        "difficulty": "hard",
        "source": "pubmedqa-pharmacogenomics-interaction",
    },
    {
        "task_input": (
            "Context: A 20-year longitudinal cohort of 4,800 adults found that "
            "shift workers had a 41% higher incidence of metabolic syndrome "
            "(HR 1.41, 95% CI 1.28–1.56). After adjusting for sleep duration, "
            "diet quality, and physical activity, the HR dropped to 1.14 "
            "(95% CI 0.99–1.31, p = 0.07). Mediation analysis suggested 73% of "
            "the crude association was explained by disrupted circadian sleep.\n"
            "Question: Does shift work independently increase metabolic syndrome risk?"
        ),
        "expected_behavior": (
            "maybe\n"
            "The crude association (HR 1.41) became non-significant after adjustment "
            "(HR 1.14, p = 0.07), and mediation analysis attributed 73% of the "
            "effect to disrupted sleep, suggesting sleep disruption rather than "
            "shift work per se is the primary driver."
        ),
        "difficulty": "hard",
        "source": "pubmedqa-mediation-confounding",
    },
    {
        "task_input": (
            "Context: A multicentre RCT of 1,100 ICU patients with septic shock "
            "compared liberal (Hb target 10 g/dL) versus restrictive (Hb target "
            "7 g/dL) transfusion strategies. 90-day mortality was 43% in both arms "
            "(p = 0.95). However, in the pre-specified subgroup with baseline "
            "SOFA ≥ 11, liberal transfusion reduced mortality (52% vs 63%, "
            "p = 0.03).\n"
            "Question: Does liberal red cell transfusion improve survival in "
            "septic shock?"
        ),
        "expected_behavior": (
            "maybe\n"
            "Overall 90-day mortality was identical (43% each, p = 0.95), but a "
            "pre-specified subgroup with high severity (SOFA ≥ 11) showed reduced "
            "mortality with liberal transfusion (52% vs 63%, p = 0.03), warranting "
            "further investigation in this subgroup."
        ),
        "difficulty": "hard",
        "source": "pubmedqa-subgroup-only-benefit",
    },
    {
        "task_input": (
            "Context: A nested case-control study within a 10-year cancer registry "
            "found that proton pump inhibitor (PPI) use > 1 year was associated "
            "with gastric cancer risk (OR 2.44, 95% CI 1.68–3.56). After excluding "
            "cases with H. pylori eradication history, the OR fell to 1.09 "
            "(95% CI 0.72–1.65). PPI prescribing volume was also highest in the "
            "same clinics that had the highest H. pylori prevalence.\n"
            "Question: Do proton pump inhibitors cause gastric cancer?"
        ),
        "expected_behavior": (
            "no\n"
            "The crude association (OR 2.44) appears to be confounded by H. pylori "
            "infection: after excluding H. pylori-eradicated cases the OR dropped "
            "to a non-significant 1.09, suggesting residual confounding rather than "
            "a causal PPI effect."
        ),
        "difficulty": "hard",
        "source": "pubmedqa-confounding-by-indication",
    },
]
