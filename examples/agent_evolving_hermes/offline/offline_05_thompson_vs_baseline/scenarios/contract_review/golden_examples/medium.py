# coding: utf-8
# Medium examples — baseline notices something is off but misses the legal doctrine
GOLDEN_EXAMPLES_MEDIUM = [
    {
        "task_input": (
            "Review this limitation of liability clause:\n\n"
            "'NEITHER PARTY SHALL BE LIABLE FOR ANY DAMAGES WHATSOEVER, INCLUDING "
            "DIRECT, INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES, ARISING FROM "
            "THIS AGREEMENT, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.'\n\n"
            "This is a paid SaaS contract worth $500,000 per year."
        ),
        "expected_behavior": (
            "Excluding ALL damages including direct damages is commercially unreasonable "
            "and may be unenforceable: courts in many jurisdictions refuse to enforce "
            "`limitation_of_liability` clauses that leave the non-breaching party with "
            "no remedy at all (`unconscionability`, `failure_of_essential_purpose`). "
            "Standard practice excludes only `consequential_damages` (lost profits, "
            "business interruption) while allowing direct damages up to an "
            "`aggregate_liability_cap` (typically 12 months of fees = $500,000 here). "
            "Carve-outs to the cap should be specified for: death/personal injury, "
            "fraud, wilful misconduct, confidentiality breach, and IP infringement. "
            "The ALL-CAPS formatting complies with `conspicuousness` requirements under "
            "UCC § 2-316 but the scope of exclusion must be narrowed."
        ),
        "difficulty": "medium",
        "source": "contract-review-limitation-all-damages",
    },
    {
        "task_input": (
            "Review this IP ownership clause:\n\n"
            "'All work product created by the Contractor under this Agreement "
            "shall be the property of the Client.'\n\n"
            "The Contractor is an individual consultant, not an employee."
        ),
        "expected_behavior": (
            "The clause relies on an implied assignment but is legally deficient: "
            "for an independent contractor (not an employee), copyright vests in the "
            "creator automatically — it does NOT transfer to the commissioning party "
            "without an explicit written assignment. The `work_made_for_hire` doctrine "
            "under US copyright law (17 U.S.C. § 101) applies only to employees or, "
            "for independent contractors, only to nine specific categories of works "
            "(contributions to collective works, motion pictures, translations, etc.). "
            "Software generally does NOT qualify. Replace with explicit language: "
            "'Contractor hereby irrevocably assigns to Client all right, title, and "
            "interest (including copyright) in all work product.' Also add a "
            "`moral_rights_waiver` for international agreements and ensure the clause "
            "carves out `pre-existing_IP` and `background_IP` of the Contractor."
        ),
        "difficulty": "medium",
        "source": "contract-review-ip-ownership",
    },
    {
        "task_input": (
            "Review this dispute resolution clause:\n\n"
            "'Any disputes arising from this Agreement shall be resolved by arbitration.'\n\n"
            "The contract is between a US company and a German company "
            "for a €2 million software integration project."
        ),
        "expected_behavior": (
            "The arbitration clause is a `pathological_arbitration_clause`: it fails "
            "to specify the arbitral institution (ICC, AAA/ICDR, LCIA, DIS), the "
            "`seat_of_arbitration` (which determines the procedural law and courts "
            "with supervisory jurisdiction), the number of arbitrators, the language "
            "of proceedings, and the substantive law. Without a seat, the clause is "
            "unenforceable in many jurisdictions. For a cross-border dispute of €2M, "
            "specify: ICC Rules (for international credibility), seat in a neutral "
            "city (London, Paris, Singapore, or Zurich), one or three arbitrators "
            "depending on complexity, English language, and that either party may "
            "seek `emergency_arbitrator` relief. Also confirm the `New_York_Convention` "
            "applies in both the US and Germany (it does) so awards are enforceable."
        ),
        "difficulty": "medium",
        "source": "contract-review-arbitration-clause",
    },
    {
        "task_input": (
            "Review this payment and offset clause:\n\n"
            "'The Client may withhold payment if it believes the Supplier has "
            "failed to perform its obligations, until such failure is remedied.'\n\n"
            "The Client is a large corporation; the Supplier is a small business."
        ),
        "expected_behavior": (
            "Unilateral and unconditional `payment_withholding` rights are a significant "
            "risk to the Supplier: 'believes' is entirely subjective, there is no "
            "dispute mechanism or cap, and the Supplier has no certainty of cash flow. "
            "This creates a `cash_flow_weapon` that a larger Client can use to force "
            "commercial concessions. Replace with: the Client may withhold only the "
            "portion of payment `directly_attributable` to the disputed failure, must "
            "give written notice specifying the alleged breach within 5 business days "
            "of the invoice, and the undisputed portion must be paid on time. Add "
            "a `set-off_waiver` for amounts below a materiality threshold, and "
            "specify that the Client's right to withhold does not arise until it has "
            "issued a `notice_to_remedy` and the `cure_period` has expired. "
            "Many jurisdictions have `prompt_payment` legislation (e.g. UK Late Payment "
            "of Commercial Debts Act) that applies regardless of contract terms."
        ),
        "difficulty": "medium",
        "source": "contract-review-payment-withholding",
    },
]
