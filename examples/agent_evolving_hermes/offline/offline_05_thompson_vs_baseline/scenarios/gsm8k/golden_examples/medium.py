# coding: utf-8
# Medium — two or three arithmetic steps; baseline gives a number but omits
# the chain of reasoning that justifies it.

GOLDEN_EXAMPLES_MEDIUM = [
    {
        "task_input": (
            "A store sells apples for $0.40 each. Jane buys 8 apples and "
            "pays with a $5 bill. How much change does she receive?"
        ),
        "expected_behavior": (
            "Cost of 8 apples: 8 × $0.40 = $3.20\n"
            "Change: $5.00 − $3.20 = $1.80\n"
            "**Answer: $1.80**"
        ),
        "difficulty": "medium",
        "source": "gsm8k-two-step-purchase",
    },
    {
        "task_input": (
            "A class has 30 students. 40% of them brought sandwiches for lunch. "
            "How many students did NOT bring a sandwich?"
        ),
        "expected_behavior": (
            "Students who brought sandwiches: 40% × 30 = 0.40 × 30 = 12\n"
            "Students without a sandwich: 30 − 12 = 18\n"
            "**Answer: 18 students**"
        ),
        "difficulty": "medium",
        "source": "gsm8k-two-step-percentage",
    },
    {
        "task_input": (
            "Mike earns $15 per hour. He works 6 hours on Monday and 8 hours "
            "on Tuesday. How much does he earn in total over the two days?"
        ),
        "expected_behavior": (
            "Monday earnings: 6 × $15 = $90\n"
            "Tuesday earnings: 8 × $15 = $120\n"
            "Total: $90 + $120 = $210\n"
            "**Answer: $210**"
        ),
        "difficulty": "medium",
        "source": "gsm8k-two-step-earnings",
    },
    {
        "task_input": (
            "A factory produces 150 units per hour. It runs 8 hours a day, "
            "5 days a week. How many units does it produce in one week?"
        ),
        "expected_behavior": (
            "Production per day: 150 × 8 = 1,200 units\n"
            "Production per week: 1,200 × 5 = 6,000 units\n"
            "**Answer: 6,000 units**"
        ),
        "difficulty": "medium",
        "source": "gsm8k-two-step-production",
    },
]
