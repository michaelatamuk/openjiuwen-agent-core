# coding: utf-8
# Medium examples — methodology issues where baseline gives "well-written" not the issue
GOLDEN_EXAMPLES_MEDIUM = [
    # ── 1. HARKing (Hypothesising After Results are Known) ───────────────────
    {
        "task_input": (
            "Review this paper:\n\n"
            "Introduction states: 'We hypothesise that anxiety will predict insomnia.'\n"
            "Results: The authors tested anxiety, depression, stress, loneliness, and "
            "social support as predictors. Only loneliness was significant (p=0.03).\n"
            "Discussion: 'As predicted, loneliness predicted insomnia, confirming our "
            "hypothesis about the social determinants of sleep.'\n\n"
            "Note: the introduction only mentioned anxiety as the hypothesis."
        ),
        "expected_behavior": (
            "Classic `HARKing` (Hypothesising After Results are Known): the introduction "
            "predicts anxiety but the discussion claims loneliness was 'predicted', "
            "retrofitting the hypothesis to the result. This inflates `Type_I_error` — "
            "with 5 predictors tested, the family-wise error rate at α=0.05 is "
            "1−(0.95)^5 ≈ 0.23. Require `pre-registration` of hypotheses, apply "
            "`Bonferroni_correction` (threshold p < 0.01), or use `FDR_correction`. "
            "The paper should acknowledge the exploratory nature of the analysis."
        ),
        "difficulty": "medium",
        "source": "paper-review-harking",
    },
    # ── 2. p-hacking through researcher degrees of freedom ────────────────────
    {
        "task_input": (
            "Review this methods section:\n\n"
            "'We collected data from 200 participants. After excluding outliers "
            "(defined as scores >2 SD from the mean), removing participants who "
            "did not complete all items, and controlling for age and gender, "
            "our final sample was n=87. The intervention effect was significant "
            "(p=0.048).'\n\n"
            "No pre-registration exists. The paper does not report analyses "
            "before exclusions or without covariates."
        ),
        "expected_behavior": (
            "Multiple `researcher_degrees_of_freedom` without pre-registration: "
            "the combination of outlier exclusion rule, item-completion threshold, "
            "and covariate selection were likely chosen post-hoc to achieve significance "
            "(`p-hacking`). Losing 113 of 200 participants (~56%) is extreme and "
            "requires justification. Report the full-sample analysis and the analysis "
            "without covariates as robustness checks. The `p-value` of 0.048 just below "
            "the threshold is a warning sign of `p-hacking`. Require `pre-registration` "
            "of the analysis pipeline or label the work as exploratory."
        ),
        "difficulty": "medium",
        "source": "paper-review-p-hacking",
    },
    # ── 3. Regression to the mean in pre-post design ──────────────────────────
    {
        "task_input": (
            "Review this study design:\n\n"
            "'Patients with severe depression (BDI > 28) received 8 weeks of "
            "cognitive behavioural therapy. Pre-treatment BDI mean = 33.4. "
            "Post-treatment BDI mean = 18.7. The 14.7-point reduction (p < 0.001) "
            "demonstrates CBT is highly effective.'\n\n"
            "The study has no control group."
        ),
        "expected_behavior": (
            "No control group means the improvement cannot be attributed to CBT. "
            "Selecting participants at extreme scores (BDI > 28) guarantees "
            "`regression_to_the_mean` — scores that are extreme at baseline will "
            "naturally move toward the population average on re-test regardless of "
            "treatment. Additional threats: `spontaneous_remission`, `maturation_effect`, "
            "and `placebo_effect`. Require a `wait-list_control` group, `randomised_controlled_trial` "
            "design, or at minimum an `active_control` receiving treatment-as-usual. "
            "Pre-post within-group designs cannot establish efficacy."
        ),
        "difficulty": "medium",
        "source": "paper-review-regression-to-mean",
    },
    # ── 4. Underpowered study claiming a null result ───────────────────────────
    {
        "task_input": (
            "Review this paper's conclusion:\n\n"
            "'We found no significant difference in recovery time between the "
            "drug and placebo groups (p=0.21, n=12 per group). This demonstrates "
            "that the drug has no effect on recovery and should not be recommended "
            "for clinical use.'\n\n"
            "The paper reports no power calculation."
        ),
        "expected_behavior": (
            "Absence of significance ≠ evidence of absence. With n=12 per group, "
            "the study has very low `statistical_power` — typically 80% power to detect "
            "a medium effect (Cohen's d=0.5) requires n≈64 per group. "
            "The `Type_II_error` (false negative) rate is likely >50%. "
            "A `power_analysis` should have been conducted a priori, and the paper should "
            "report `confidence_intervals` for the effect size to show what magnitudes "
            "of effect are compatible with the data. The conclusion 'has no effect' is "
            "unjustified; write 'insufficient power to detect a clinically meaningful effect'."
        ),
        "difficulty": "medium",
        "source": "paper-review-underpowered-null",
    },
    # ── 5. Convenience sample over-generalised ────────────────────────────────
    {
        "task_input": (
            "Review this paper's discussion:\n\n"
            "'Our findings that social media increases anxiety are generalisable to "
            "the global population, providing strong evidence for regulatory action.'\n\n"
            "Methods: 'Participants were 94 undergraduate psychology students at a "
            "single US university who received course credit for participation.'"
        ),
        "expected_behavior": (
            "`WEIRD sample` problem (`Western, Educated, Industrialised, Rich, Democratic`): "
            "undergraduate psychology students are not representative of the global population. "
            "`Convenience_sampling` with course-credit incentive introduces `volunteer_bias`. "
            "The claim of global generalisability is unjustified. The paper should characterise "
            "the sample's limits in the Limitations section, restrict conclusions to similar "
            "populations, and avoid calling for regulatory action from a single convenience "
            "sample. `External_validity` requires replication across diverse populations."
        ),
        "difficulty": "medium",
        "source": "paper-review-weird-sample",
    },
    # ── 6. Mediation analysis without temporal ordering ───────────────────────
    {
        "task_input": (
            "Review this mediation analysis:\n\n"
            "'We tested whether self-efficacy mediates the relationship between "
            "training and performance. Training → self-efficacy → performance "
            "(indirect effect β=0.23, 95% CI [0.08, 0.41]). All variables were "
            "measured simultaneously in a single cross-sectional survey.'\n\n"
        ),
        "expected_behavior": (
            "`Cross-sectional mediation` is causally unidentified: mediation requires "
            "temporal ordering (cause precedes mediator precedes outcome), but measuring "
            "all three variables at the same time means any causal direction is equally "
            "plausible (performance could cause self-efficacy, self-efficacy could cause "
            "training, etc.). `Bootstrapped confidence intervals` do not solve this — they "
            "only address sampling variability, not causal identification. "
            "Require a `longitudinal design` with at least three time points (T1: training, "
            "T2: self-efficacy, T3: performance) or an `experimental manipulation` of "
            "the mediator to establish causal mediation."
        ),
        "difficulty": "medium",
        "source": "paper-review-cross-sectional-mediation",
    },
    # ── 7. WEIRD + non-equivalent groups comparison ───────────────────────────
    {
        "task_input": (
            "Review this study:\n\n"
            "'We compared depression scores between participants who exercise regularly "
            "(n=45, mean BDI=12.3) and those who do not (n=41, mean BDI=19.8). "
            "Regular exercisers had significantly lower depression (t=4.2, p<0.001). "
            "We conclude that exercise reduces depression.'\n\n"
            "Groups were not randomly assigned; participants self-reported exercise habits."
        ),
        "expected_behavior": (
            "Non-equivalent groups from self-selection: regular exercisers likely differ "
            "from non-exercisers on many variables beyond exercise (diet, sleep, social "
            "support, socioeconomic status, baseline health) — these are `confounding_variables` "
            "that could explain the BDI difference. This is a `quasi-experimental design`, "
            "not an experiment; causal language ('reduces depression') is unwarranted. "
            "Require `propensity_score_matching` or `regression_discontinuity` to control "
            "for observed confounders, or acknowledge the correlational nature and use "
            "'associated with' language. The study cannot rule out `reverse_causality` "
            "(depressed people exercise less)."
        ),
        "difficulty": "medium",
        "source": "paper-review-self-selection-confound",
    },
    # ── 8. Selective citation / publication bias ──────────────────────────────
    {
        "task_input": (
            "Review this literature review section:\n\n"
            "'Numerous studies confirm that open-plan offices increase collaboration "
            "and productivity (Jones 2018; Smith 2019; Chen 2020; Park 2021). "
            "Based on this consistent evidence, we hypothesise that open-plan design "
            "will improve team performance.'\n\n"
            "A quick search reveals 12 studies showing the opposite effect "
            "that are not cited."
        ),
        "expected_behavior": (
            "`Selective citation` and likely `publication_bias`: citing only supporting "
            "studies while ignoring contradictory evidence creates a misleading picture "
            "of the field. A proper literature review requires a `systematic_search` "
            "following `PRISMA` guidelines with explicit inclusion/exclusion criteria, "
            "or at minimum acknowledgement of conflicting findings. The `file_drawer_problem` "
            "(null results are less likely to be published) means even an honest search "
            "may overrepresent positive findings. The hypothesis should acknowledge the "
            "mixed evidence rather than citing a 'consistent' literature that does not exist."
        ),
        "difficulty": "medium",
        "source": "paper-review-publication-bias",
    },
]
