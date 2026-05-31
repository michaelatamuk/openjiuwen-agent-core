# coding: utf-8
# Hard examples — baseline says "well-written, findings clearly presented"; judge ≈ 0.05
GOLDEN_EXAMPLES_HARD = [
    # ── 1. p-hacking via optional stopping ────────────────────────────────────
    {
        "task_input": (
            "Review this methods section:\n\n"
            "'Data collection continued until we reached statistical significance. "
            "We began with n=20, added 10 participants per batch, and stopped at "
            "n=60 when p=0.043 was achieved. This adaptive approach is efficient "
            "and ensures the study is adequately powered.'\n\n"
        ),
        "expected_behavior": (
            "`Optional_stopping` (peeking at data and stopping when p<0.05) severely "
            "inflates the `Type_I_error` rate. With three interim analyses at n=20, 40, 60, "
            "the true α can exceed 0.14 even if no effect exists (O'Brien-Fleming). "
            "This is not adaptive design — it is uncontrolled `p-hacking`. Legitimate "
            "`sequential_analysis` requires pre-specified stopping rules with alpha-spending "
            "functions (e.g. `O'Brien-Fleming` or `Pocock` boundaries) that adjust the "
            "significance threshold at each look to maintain overall α=0.05. "
            "Alternatively, a `Bayesian adaptive design` with pre-specified priors is valid. "
            "The study must be reanalysed with corrected thresholds or pre-registered "
            "stopping rules."
        ),
        "difficulty": "hard",
        "source": "paper-review-optional-stopping",
    },
    # ── 2. Measurement invariance not tested before group comparison ───────────
    {
        "task_input": (
            "Review this study:\n\n"
            "'We compared trait anxiety (measured with the STAI) between men (n=120) "
            "and women (n=130). Women scored significantly higher (M=44.2 vs M=38.7, "
            "p<0.001, d=0.58). This confirms that women experience more anxiety than men.'\n\n"
            "No measurement invariance testing is reported."
        ),
        "expected_behavior": (
            "Comparing latent construct scores across groups without testing "
            "`measurement_invariance` is methodologically flawed. If the STAI items "
            "function differently across gender (e.g. different factor loadings or item "
            "intercepts), the observed mean difference reflects measurement artefacts, "
            "not true trait differences. The analysis requires `confirmatory_factor_analysis` "
            "with a sequence of nested models: configural → metric → scalar invariance. "
            "Only scalar invariance (equal loadings AND intercepts) justifies latent mean "
            "comparisons. Using `lavaan` or similar SEM software, report chi-square "
            "difference tests and `CFI` change (ΔCFI < 0.01 is acceptable). Without this, "
            "the group comparison is uninterpretable."
        ),
        "difficulty": "hard",
        "source": "paper-review-measurement-invariance",
    },
    # ── 3. Spurious correlation from shared method variance ────────────────────
    {
        "task_input": (
            "Review this study:\n\n"
            "'Employee job satisfaction and organisational commitment were measured "
            "using self-report questionnaires in a single survey session. "
            "The correlation between the two constructs was r=0.71 (p<0.001), "
            "indicating a strong positive relationship.'\n\n"
        ),
        "expected_behavior": (
            "Strong correlation between two self-report measures collected simultaneously "
            "is likely inflated by `common_method_variance` (`CMV`): response tendencies "
            "(acquiescence bias, social desirability, mood at time of survey) affect both "
            "measures in the same direction. `Harman's_single-factor_test` is a minimal "
            "check; stronger remedies include temporal separation of measurements, "
            "using different response formats (Likert vs. semantic differential), "
            "including a marker variable, or collecting at least one construct from an "
            "objective source (e.g. supervisor ratings). The correlation r=0.71 likely "
            "overestimates the true construct-level relationship. Report `discriminant_validity` "
            "via the `AVE` (average variance extracted) compared to the squared inter-construct "
            "correlation."
        ),
        "difficulty": "hard",
        "source": "paper-review-common-method-variance",
    },
    # ── 4. Inflated effect size from small-study bias / winner's curse ─────────
    {
        "task_input": (
            "Review this meta-analysis section:\n\n"
            "'We identified 8 studies of the intervention. The pooled effect size "
            "was d=0.82 (95% CI [0.61, 1.03]), indicating a large effect. "
            "Heterogeneity was low (I²=18%). The funnel plot was not examined "
            "because I² was low, indicating consistent results.'\n\n"
        ),
        "expected_behavior": (
            "Three errors: (1) Only 8 studies — `funnel_plot_asymmetry` testing "
            "(Egger's test, Begg's test) is meaningful with ≥10 studies, but the "
            "funnel plot should still be examined visually and small-study effects "
            "checked with `trim-and-fill`. Low `I²` does not rule out `publication_bias` "
            "— it only describes between-study variability. (2) `Winner's curse`: "
            "small positive studies are more likely to be published (`publication_bias`), "
            "inflating pooled estimates. (3) d=0.82 is very large for a behavioural "
            "intervention — a `p-curve` analysis should check whether the distribution "
            "of p-values is right-skewed (indicative of genuine effect) or flat/left-skewed "
            "(indicative of p-hacking). Report `fail-safe_N` (Rosenthal's method) and "
            "conduct sensitivity analysis excluding the smallest studies."
        ),
        "difficulty": "hard",
        "source": "paper-review-meta-analysis-bias",
    },
    # ── 5. Latent growth curve model misspecification ─────────────────────────
    {
        "task_input": (
            "Review this longitudinal analysis:\n\n"
            "'We modelled change in depression over 4 time points using a "
            "latent growth curve model (LGCM). The linear slope was significant "
            "(β=-0.31, p<0.001), indicating continuous improvement over time. "
            "Model fit: CFI=0.94, RMSEA=0.08.'\n\n"
        ),
        "expected_behavior": (
            "The linear LGCM fit is marginal: `RMSEA`=0.08 is at the upper bound of "
            "acceptable fit (criterion < 0.06–0.08) and `CFI`=0.94 is below the recommended "
            "0.95 threshold. With only 4 time points, a quadratic or piecewise growth "
            "model should be compared using `likelihood_ratio_test` (chi-square difference) "
            "and `AIC`/`BIC`. Depression trajectories are often non-linear (rapid initial "
            "improvement then plateau), so forcing linearity may misrepresent recovery. "
            "Also report: intercept variance (do people start at different levels?), "
            "slope variance (do people change at different rates?), and intercept-slope "
            "covariance (do higher-baseline people improve faster or slower?). "
            "These random effects are the key substantive findings of an LGCM."
        ),
        "difficulty": "hard",
        "source": "paper-review-lgcm-fit",
    },
    # ── 6. Omitted variable bias in regression ───────────────────────────────
    {
        "task_input": (
            "Review this regression analysis:\n\n"
            "'We regressed employee turnover intention on pay satisfaction (β=-0.42, p<0.001) "
            "and job autonomy (β=-0.29, p<0.001). Together these explain 31% of the variance "
            "(R²=0.31). The results demonstrate that increasing pay and autonomy will "
            "substantially reduce turnover.'\n\n"
            "The model does not include supervisor quality, career development, "
            "or organisational culture variables."
        ),
        "expected_behavior": (
            "`Omitted_variable_bias` (`OVB`): supervisor quality, career development, and "
            "organisational culture are well-established predictors of turnover intention "
            "that correlate with both pay satisfaction and the outcome. Their omission "
            "likely inflates the coefficients on the included predictors. "
            "R²=0.31 leaving 69% of variance unexplained suggests the model is incomplete. "
            "A `DAG` (directed acyclic graph) should map the causal structure before "
            "selecting controls. The causal interpretation ('will substantially reduce') "
            "requires controlling for all `backdoor paths` from pay/autonomy to turnover. "
            "Conduct sensitivity analysis for unobserved confounding using `E-value` "
            "(Ding & VanderWeele) to report how strong an unmeasured confounder would "
            "need to be to explain away the association."
        ),
        "difficulty": "hard",
        "source": "paper-review-omitted-variable-bias",
    },
    # ── 7. NHST + equivalence testing confusion ───────────────────────────────
    {
        "task_input": (
            "Review this conclusion:\n\n"
            "'The new low-cost intervention did not differ significantly from the "
            "gold-standard treatment (t(58)=1.42, p=0.16). As the treatments are "
            "equivalent, we recommend adopting the cheaper alternative in clinical practice.'\n\n"
        ),
        "expected_behavior": (
            "Fundamental `NHST` misuse: failure to reject H₀ (p=0.16) is not evidence "
            "of equivalence — it is simply insufficient evidence of difference, which may "
            "be due to low `statistical_power` rather than true equivalence. "
            "Claiming clinical equivalence requires `equivalence_testing` (TOST procedure): "
            "pre-specify a smallest effect size of interest (SESOI, e.g. d=0.3), then test "
            "whether the `90%_confidence_interval` of the effect falls entirely within "
            "[−SESOI, +SESOI]. Report the observed effect size and its confidence interval "
            "to show what magnitudes of difference are compatible with the data. "
            "The recommendation to replace standard treatment based solely on p=0.16 "
            "from an underpowered study is clinically irresponsible."
        ),
        "difficulty": "hard",
        "source": "paper-review-equivalence-testing",
    },
    # ── 8. Network meta-analysis transitivity assumption violation ─────────────
    {
        "task_input": (
            "Review this network meta-analysis:\n\n"
            "'We combined direct and indirect evidence across 23 trials to rank "
            "8 antidepressants. Drug A ranked first (SUCRA=0.89). The network "
            "was connected through a common comparator (placebo). "
            "Inconsistency was assessed globally (Q=12.4, df=9, p=0.19).'\n\n"
        ),
        "expected_behavior": (
            "Two issues: (1) Global inconsistency test (Q) is low power with sparse networks "
            "— local inconsistency should also be evaluated per loop using the `node-splitting` "
            "method or `design-by-treatment_interaction` model. (2) `Transitivity assumption`: "
            "combining trials through placebo requires that effect modifiers (patient severity, "
            "duration, concomitant medications) are similarly distributed across trial designs — "
            "this should be verified by tabulating `treatment_effect_modifiers` across the "
            "network. `SUCRA` rankings have wide uncertainty intervals; report `rankograms` "
            "and the `SUCRA` 95% CrI to show ranking uncertainty, not just point estimates. "
            "The ranking Drug A = best may flip entirely with sensitivity analysis excluding "
            "older trials with higher placebo response rates."
        ),
        "difficulty": "hard",
        "source": "paper-review-network-meta-analysis",
    },
]
