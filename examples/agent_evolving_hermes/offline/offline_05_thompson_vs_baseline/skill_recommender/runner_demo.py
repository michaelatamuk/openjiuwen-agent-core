from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .recommender import build_recommender
from .runner_printer import _print_query_results

_DEMO_MATRIX = [
    {
        "run_id":          "demo_smarthub_001",
        "skill_name":      "smarthub-support",
        "fitness_metrics": ["bag_of_words", "f1", "format"],
        "baseline_cross_eval": [
            {
                "example_id":       "sh_01",
                "example_input":    "How do I reset my SmartHub device to factory settings?",
                "example_expected": "Hold the reset button for 10 seconds until the LED flashes red.",
                "candidate_output": "Press and hold the reset button on the back for 10 seconds.",
                "scores": {"bag_of_words": 0.82, "f1": 0.74, "format": 0.90},
            },
            {
                "example_id":       "sh_02",
                "example_input":    "My SmartHub keeps disconnecting from WiFi, what should I do?",
                "example_expected": "Restart the hub and check that the firmware is up to date.",
                "candidate_output": "Try restarting the hub. Also make sure the firmware is current.",
                "scores": {"bag_of_words": 0.78, "f1": 0.71, "format": 0.85},
            },
            {
                "example_id":       "sh_03",
                "example_input":    "How do I add a new device to my SmartHub network?",
                "example_expected": "Open the SmartHub app and tap 'Add Device', then follow the prompts.",
                "candidate_output": "In the SmartHub app, go to Add Device and follow the on-screen steps.",
                "scores": {"bag_of_words": 0.88, "f1": 0.81, "format": 0.92},
            },
            {
                "example_id":       "sh_04",
                "example_input":    "What does the blinking orange light on my hub mean?",
                "example_expected": "Blinking orange means the hub is searching for an internet connection.",
                "candidate_output": "A blinking orange LED indicates the hub is trying to connect to the internet.",
                "scores": {"bag_of_words": 0.80, "f1": 0.76, "format": 0.88},
            },
            {
                "example_id":       "sh_05",
                "example_input":    "Can I connect my SmartHub to a 5GHz WiFi network?",
                "example_expected": "Yes, the SmartHub supports both 2.4GHz and 5GHz bands.",
                "candidate_output": "The SmartHub supports both 2.4 GHz and 5 GHz wireless networks.",
                "scores": {"bag_of_words": 0.75, "f1": 0.69, "format": 0.80},
            },
        ],
    },
    {
        "run_id":          "demo_billing_001",
        "skill_name":      "billing-support",
        "fitness_metrics": ["bag_of_words", "f1", "format"],
        "baseline_cross_eval": [
            {
                "example_id":       "bi_01",
                "example_input":    "How do I update my payment method?",
                "example_expected": "Log in, go to Account → Billing, then click Change Payment Method.",
                "candidate_output": "Sign in and navigate to Account > Billing > Change Payment Method.",
                "scores": {"bag_of_words": 0.84, "f1": 0.79, "format": 0.91},
            },
            {
                "example_id":       "bi_02",
                "example_input":    "Why was I charged twice this month?",
                "example_expected": "This can happen if two subscriptions overlap. Contact support to resolve.",
                "candidate_output": "Double charges usually mean overlapping subscriptions. Please contact us.",
                "scores": {"bag_of_words": 0.70, "f1": 0.65, "format": 0.78},
            },
            {
                "example_id":       "bi_03",
                "example_input":    "How do I download my invoice as a PDF?",
                "example_expected": "Go to Account → Billing History and click the PDF icon next to any invoice.",
                "candidate_output": "In Account > Billing History you can click the PDF icon to download invoices.",
                "scores": {"bag_of_words": 0.86, "f1": 0.80, "format": 0.93},
            },
            {
                "example_id":       "bi_04",
                "example_input":    "What payment methods do you accept?",
                "example_expected": "We accept Visa, Mastercard, Amex, PayPal, and bank transfer.",
                "candidate_output": "Accepted payment methods: Visa, Mastercard, Amex, PayPal, and bank transfer.",
                "scores": {"bag_of_words": 0.91, "f1": 0.87, "format": 0.95},
            },
            {
                "example_id":       "bi_05",
                "example_input":    "Can I get a refund for my last subscription payment?",
                "example_expected": "Refunds are available within 14 days of payment. Submit a refund request.",
                "candidate_output": "If the payment was made within 14 days, you can request a refund.",
                "scores": {"bag_of_words": 0.77, "f1": 0.72, "format": 0.82},
            },
        ],
    },
]

_DEMO_QUERIES = [
    "My router keeps dropping the connection, how do I fix it?",
    "I need a PDF copy of my latest invoice",
    "What lights should I see when the hub is working normally?",
]

def _run_demo() -> None:
    """Synthetic quick-start demo — no network or GEPA run required."""
    with tempfile.TemporaryDirectory(prefix="skill_recommender_demo_") as tmpdir:
        oracle_dir = Path(tmpdir)

        for entry in _DEMO_MATRIX:
            fname = f"scoring_matrix_{entry['run_id']}.json"
            with open(oracle_dir / fname, "w", encoding="utf-8") as fh:
                json.dump(entry, fh, indent=2, ensure_ascii=False)

        print(f"\n  Demo matrix written to: {oracle_dir}")
        rec = build_recommender(oracle_dir=oracle_dir, variant="baseline", embedder_method="tfidf")
        print(f"  Loaded  : {rec.n_examples} rows · "
              f"{len(rec.skills)} skill(s) · {len(rec.metrics)} metric(s)")
        print(f"  Skills  : {rec.skills}")
        print(f"  Metrics : {rec.metrics}")

        for query in _DEMO_QUERIES:
            results = rec.recommend(query=query, sim_threshold=0.10,
                                    score_threshold=0.15, top_k=3)
            _print_query_results(results, query, "baseline")
