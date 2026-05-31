# coding: utf-8
# Easy examples — obvious structural gaps any reviewer would notice
GOLDEN_EXAMPLES_EASY = [
    {
        "task_input": (
            "Review this contract clause:\n\n"
            "'The Supplier will deliver the goods. Payment will be made by the Client.'\n\n"
            "No amount, currency, or due date is specified anywhere in the contract."
        ),
        "expected_behavior": (
            "Payment terms are incomplete: the contract must state (1) the exact amount "
            "or pricing formula, (2) currency, (3) payment due date or milestone trigger, "
            "and (4) the accepted payment method. Without these, the obligation is too "
            "vague to be an enforceable `payment_obligation`. Add: 'The Client shall pay "
            "the Supplier USD $X within 30 days of invoice date.' Also specify whether "
            "`late_payment_interest` applies (e.g. 1.5 % per month on overdue sums) "
            "and who bears bank transfer fees."
        ),
        "difficulty": "easy",
        "source": "contract-review-payment-terms",
    },
    {
        "task_input": (
            "Review this contract:\n\n"
            "'This Agreement is made between ABC Corp and XYZ Ltd. "
            "The parties agree to collaborate on a marketing project.'\n\n"
            "No governing law clause or dispute resolution clause is present anywhere "
            "in the 12-page agreement."
        ),
        "expected_behavior": (
            "A `governing_law` clause is missing — without it, courts must determine "
            "applicable law from circumstantial factors (place of performance, "
            "domicile of parties), which is unpredictable and expensive. Add: "
            "'This Agreement shall be governed by and construed in accordance with "
            "the laws of [State/Country], without regard to its conflict of law principles.' "
            "Also add a `dispute_resolution` clause specifying whether disputes go to "
            "`litigation` (name the court and waive objection to jurisdiction) or "
            "`arbitration` (specify rules, seat, language, and number of arbitrators). "
            "Without these, enforcing the contract across jurisdictions is uncertain."
        ),
        "difficulty": "easy",
        "source": "contract-review-governing-law",
    },
    {
        "task_input": (
            "Review this confidentiality clause:\n\n"
            "'Each party agrees to keep the other party's information confidential "
            "and not to disclose it to third parties.'\n\n"
            "The contract is a 3-year software development agreement."
        ),
        "expected_behavior": (
            "The `confidentiality_clause` is incomplete in three ways: (1) No duration — "
            "does the obligation survive termination, and for how long? Standard practice "
            "is 2–5 years post-termination or perpetual for trade secrets. (2) No "
            "`return_or_destruction` obligation — at termination the recipient should "
            "be required to return or certify destruction of confidential materials "
            "within a specified period. (3) 'Third parties' needs carve-outs: employees, "
            "advisors, and subcontractors who need access should be permitted "
            "subject to equivalent `confidentiality_obligations` flowing down."
        ),
        "difficulty": "easy",
        "source": "contract-review-confidentiality",
    },
    {
        "task_input": (
            "Review this termination clause:\n\n"
            "'Either party may terminate this Agreement at any time.'\n\n"
            "The contract is for ongoing SaaS services and has no notice period "
            "or exit provisions."
        ),
        "expected_behavior": (
            "Termination at any time without notice is commercially unreasonable: "
            "the service provider needs time to wind down operations and the client "
            "needs time to migrate data. Add: (1) a `notice_period` for "
            "`termination_for_convenience` (typically 30–90 days written notice); "
            "(2) `termination_for_cause` allowing immediate termination for material "
            "breach, subject to a `cure_period` (typically 30 days after written notice); "
            "(3) `consequences_of_termination`: data export window, transition assistance "
            "obligations, survival of payment and confidentiality clauses. "
            "Without these, the clause creates `contractual_uncertainty` for both parties."
        ),
        "difficulty": "easy",
        "source": "contract-review-termination",
    },
]
