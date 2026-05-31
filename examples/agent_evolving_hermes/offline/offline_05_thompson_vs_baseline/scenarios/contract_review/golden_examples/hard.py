# coding: utf-8
# Hard examples — baseline says "clause present, terms seem clear"; judge ≈ 0.05
GOLDEN_EXAMPLES_HARD = [
    # ── 1. Liquidated damages clause as unenforceable penalty ─────────────────
    {
        "task_input": (
            "Review this clause from a construction contract:\n\n"
            "'If the Contractor fails to complete the works by the Completion Date, "
            "the Contractor shall pay the Client £50,000 for each week of delay, "
            "as compensation for the inconvenience caused.'\n\n"
            "The total contract value is £800,000. The Client's actual losses from "
            "a one-week delay are estimated at £8,000 in lost rental income."
        ),
        "expected_behavior": (
            "This is an unenforceable `penalty_clause`, not a valid `liquidated_damages` "
            "clause. Under English law (Cavendish Square v Makdessi [2015] UKSC 67), "
            "a damages clause is a penalty — and therefore void — if it is not a "
            "`genuine_pre-estimate_of_loss` at the time of contracting AND it imposes "
            "a detriment on the breaching party out of proportion to the innocent "
            "party's legitimate interest. £50,000/week against £8,000 actual loss is "
            "a 6x multiplier with no commercial justification. Courts will strike the "
            "clause and leave the Client to prove `general_damages`, which may recover "
            "only the actual £8,000. Replace with a rate anchored to documented loss: "
            "£8,000–10,000 per week, supported by a loss estimate schedule in an "
            "appendix. Ensure the clause includes a `cap_on_liquidated_damages` "
            "(typically 10–15% of contract value) and states that LD are the Client's "
            "sole remedy for delay (preventing concurrent claims)."
        ),
        "difficulty": "hard",
        "source": "contract-review-liquidated-damages",
    },
    # ── 2. Force majeure: ejusdem generis trap ────────────────────────────────
    {
        "task_input": (
            "Review this force majeure clause:\n\n"
            "'Neither party shall be liable for failure to perform its obligations "
            "if such failure is caused by fire, flood, earthquake, hurricane, "
            "or other events beyond the party's reasonable control.'\n\n"
            "The contract is a 5-year supply agreement signed in 2024."
        ),
        "expected_behavior": (
            "The phrase 'or other events' is fatally limited by `ejusdem_generis`: "
            "a common law canon of construction that restricts a general phrase following "
            "a list of specific items to things of the same kind. Here the list is "
            "exclusively natural disasters, so 'other events' is likely interpreted "
            "as natural disasters only — excluding pandemic/epidemic, government-mandated "
            "shutdowns, cyberattack, port strikes, supplier insolvency, and war (unless "
            "a nuclear attack counts as a natural event). Post-COVID this is a known "
            "litigation risk. Replace the general tail with an explicit list: "
            "'pandemic or epidemic declared by a government or WHO authority, "
            "government-ordered lockdown or trade restriction, `cyberattack` on "
            "critical infrastructure, acts of war or terrorism, labour disputes "
            "not involving the affected party's own workforce.' Also require the "
            "invoking party to give written notice within 5 business days and use "
            "`commercially_reasonable_efforts` to mitigate."
        ),
        "difficulty": "hard",
        "source": "contract-review-force-majeure-ejusdem-generis",
    },
    # ── 3. IP indemnification carved out from liability cap ───────────────────
    {
        "task_input": (
            "Review these two clauses from a software licence agreement:\n\n"
            "Clause 12 (Limitation of Liability): 'Vendor's total aggregate liability "
            "under this Agreement shall not exceed £100,000.'\n\n"
            "Clause 13 (Indemnification): 'Vendor shall indemnify Client against all "
            "losses, damages, and legal costs arising from any claim that the Software "
            "infringes a third party's intellectual property rights.'\n\n"
        ),
        "expected_behavior": (
            "The interaction between Clauses 12 and 13 creates unlimited `IP_indemnification` "
            "exposure. Clause 13 is a separate indemnification obligation with no monetary "
            "cap — it is NOT subject to the £100,000 `aggregate_liability_cap` in Clause 12 "
            "unless Clause 12 explicitly states it applies to indemnification obligations. "
            "A single IP infringement claim (e.g. a patent assertion entity targeting the "
            "Software) can generate millions in legal costs and damages, all of which "
            "fall on the Vendor with no ceiling. To fix: either (a) bring IP indemnification "
            "within the cap, or (b) set a separate, higher `IP_indemnification_cap` "
            "(e.g. 2× or 5× annual fees) with a `carve_out` for wilful infringement "
            "which remains unlimited. Also add Vendor's standard IP remedies: modify "
            "the Software to be non-infringing, procure a licence, or refund fees as "
            "last resort (`step-in_rights`)."
        ),
        "difficulty": "hard",
        "source": "contract-review-ip-cap-carve-out",
    },
    # ── 4. Anti-assignment clause without change-of-control carve-out ─────────
    {
        "task_input": (
            "Review this assignment clause:\n\n"
            "'Neither party may assign, transfer, or delegate any of its rights or "
            "obligations under this Agreement without the prior written consent of "
            "the other party, which shall not be unreasonably withheld.'\n\n"
            "The contract is a 10-year exclusive distribution agreement. "
            "The Distributor is being acquired by a private equity firm."
        ),
        "expected_behavior": (
            "The `anti-assignment` clause lacks a `change_of_control` carve-out, "
            "creating an M&A veto right. In most jurisdictions, a merger or acquisition "
            "constitutes a `deemed_assignment` that triggers this clause — meaning the "
            "Supplier can block or renegotiate the acquisition by withholding consent. "
            "Even 'not unreasonably withheld' gives the Supplier leverage to extract "
            "concessions (price renegotiation, security deposit, shortened exclusivity). "
            "The acquiring PE firm should require: (a) a carve-out permitting assignment "
            "to any `successor_entity` or `affiliate` without consent in a `change_of_control` "
            "transaction, OR (b) a `novation` mechanism with a defined consent timeline "
            "(e.g. deemed consent if no response within 20 business days). "
            "Also check whether the clause applies to `collateral_assignment` for "
            "lender security purposes — PE acquisition financing typically requires "
            "assignment of material contracts as security."
        ),
        "difficulty": "hard",
        "source": "contract-review-anti-assignment-coc",
    },
    # ── 5. Auto-renewal with tight notice window ──────────────────────────────
    {
        "task_input": (
            "Review this renewal clause:\n\n"
            "'This Agreement shall automatically renew for successive one-year terms "
            "unless either party provides written notice of non-renewal at least "
            "90 days before the end of the then-current term. The annual fee for "
            "each renewal term shall increase by 8% over the prior year.'\n\n"
            "The contract is a £200,000/year enterprise software agreement."
        ),
        "expected_behavior": (
            "Three compounding risks in this `auto-renewal` (`evergreen_clause`): "
            "(1) The 90-day `termination_window` is unusually long — most enterprise "
            "agreements use 30–60 days. Missing the window by one day locks the Client "
            "into another year at a higher price. The Client should negotiate this down "
            "to 30 days and require the Vendor to send a `renewal_reminder_notice` "
            "at least 120 days before the window opens. (2) An 8% annual price increase "
            "compounds aggressively: over 5 renewal cycles the fee rises from £200k to "
            "~£293k. Cap increases at CPI or a fixed lower percentage (3–4%). "
            "(3) The renewal is 'successive' with no maximum — the contract could run "
            "indefinitely. Add a `maximum_term` (e.g. 5 years total) after which "
            "the parties must actively re-sign. Without protections, the `auto-renewal` "
            "mechanism disproportionately favours the Vendor."
        ),
        "difficulty": "hard",
        "source": "contract-review-auto-renewal-trap",
    },
    # ── 6. Warranty disclaimer missing UCC magic words ────────────────────────
    {
        "task_input": (
            "Review this warranty clause from a software sale agreement governed by "
            "the laws of New York:\n\n"
            "'THE VENDOR DISCLAIMS ALL WARRANTIES TO THE MAXIMUM EXTENT PERMITTED "
            "BY APPLICABLE LAW.'\n\n"
        ),
        "expected_behavior": (
            "The disclaimer is legally insufficient under UCC § 2-316 (applicable in "
            "New York for software sale transactions). To exclude the implied warranty "
            "of `merchantability`, the disclaimer must mention the word 'merchantability' "
            "explicitly and, if in writing, must be `conspicuous` (the ALL-CAPS "
            "satisfies conspicuousness). To exclude the implied warranty of "
            "`fitness_for_a_particular_purpose`, the disclaimer must be in writing and "
            "conspicuous — but courts also require it to be specific. 'To the maximum "
            "extent permitted by law' without naming these warranties is routinely held "
            "ineffective. Replace with: 'THE VENDOR DISCLAIMS ALL WARRANTIES, EXPRESS "
            "OR IMPLIED, INCLUDING THE IMPLIED WARRANTIES OF `MERCHANTABILITY`, "
            "`FITNESS_FOR_A_PARTICULAR_PURPOSE`, AND NON-INFRINGEMENT. THE SOFTWARE "
            "IS PROVIDED AS IS.' Also check whether the `CISG` (UN Convention on "
            "Contracts for the International Sale of Goods) needs to be expressly "
            "excluded if either party is outside the US."
        ),
        "difficulty": "hard",
        "source": "contract-review-ucc-warranty-disclaimer",
    },
    # ── 7. Non-compete with overbroad scope ───────────────────────────────────
    {
        "task_input": (
            "Review this non-compete clause from a senior marketing manager's "
            "employment contract in California:\n\n"
            "'For a period of 3 years after termination of employment, Employee "
            "shall not, directly or indirectly, engage in any business that competes "
            "with the Company anywhere in the world.'\n\n"
        ),
        "expected_behavior": (
            "This `non-compete` is void and unenforceable in California under "
            "Business & Professions Code § 16600, which prohibits virtually all "
            "post-employment non-competes regardless of scope or duration — there is "
            "no `blue-penciling` (courts will not narrow it; they void it entirely). "
            "AB 2288 (effective 2024) also prohibits including void non-compete clauses "
            "in contracts, with penalties for employers who attempt to enforce them. "
            "Even in states that permit `non-compete` agreements, 3 years worldwide "
            "for a marketing manager would typically fail the `reasonableness` test: "
            "courts assess `temporal_scope` (courts rarely uphold >12–18 months), "
            "`geographic_scope` (worldwide is almost always overbroad for a non-executive), "
            "and `legitimate_business_interest` (protecting trade secrets and client "
            "relationships, not merely preventing competition). Replace with narrow "
            "`non-solicitation_of_clients` and `non-solicitation_of_employees` clauses, "
            "which are enforceable in California and most US states."
        ),
        "difficulty": "hard",
        "source": "contract-review-non-compete",
    },
    # ── 8. One-sided indemnification covering indemnitee's own negligence ─────
    {
        "task_input": (
            "Review this indemnification clause from an IT services agreement:\n\n"
            "'The Service Provider shall defend, indemnify, and hold harmless the "
            "Client and its officers, directors, employees, and agents from and against "
            "any and all claims, damages, losses, and expenses (including legal fees) "
            "arising out of or relating to the Service Provider's performance of "
            "services under this Agreement.'\n\n"
        ),
        "expected_behavior": (
            "The phrase 'arising out of or relating to the Service Provider's performance' "
            "is dangerously overbroad and likely requires the Service Provider to indemnify "
            "the Client for the Client's OWN negligence. 'Arising out of performance' "
            "includes scenarios where the Client contributed to or caused the loss: "
            "e.g. a data breach caused 80% by Client's misconfigured firewall and 20% "
            "by Service Provider. Under this clause, Service Provider indemnifies 100% "
            "of the loss. Courts in most US states and England will enforce "
            "`indemnification_for_indemnitee_negligence` only if the contract "
            "expresses that intent in clear and unequivocal language — but enforce it "
            "they will. Replace with: 'arising out of or resulting from the Service "
            "Provider's negligence, wilful misconduct, or material breach, but excluding "
            "losses to the extent caused by the Client's own negligence or wilful "
            "misconduct' (`comparative_fault` carve-out). Add `mutual_indemnification` "
            "so the Client indemnifies the Service Provider for Client-caused harm, "
            "and ensure indemnification is subject to the `aggregate_liability_cap`."
        ),
        "difficulty": "hard",
        "source": "contract-review-one-sided-indemnification",
    },
]
