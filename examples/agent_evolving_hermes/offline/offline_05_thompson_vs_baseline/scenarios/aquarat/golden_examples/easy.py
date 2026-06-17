# coding: utf-8
# Easy — single algebraic equation; baseline may estimate the right ballpark
# but will not show working or select the correct lettered option.

GOLDEN_EXAMPLES_EASY = [
    {
        "task_input": (
            "A shopkeeper bought 40 items at $5 each and sold all of them at "
            "$8 each. What is the total profit?\n"
            "Options:\n"
            "A) $100\n"
            "B) $120\n"
            "C) $150\n"
            "D) $200\n"
            "E) $80"
        ),
        "expected_behavior": (
            "Cost price = 40 × $5 = $200\n"
            "Selling price = 40 × $8 = $320\n"
            "Profit = $320 − $200 = $120\n"
            "Answer: B"
        ),
        "difficulty": "easy",
        "source": "aquarat-profit-calculation",
    },
    {
        "task_input": (
            "A train travels at 60 km/h. How long does it take to cover 180 km?\n"
            "Options:\n"
            "A) 2 hours\n"
            "B) 2.5 hours\n"
            "C) 3 hours\n"
            "D) 3.5 hours\n"
            "E) 4 hours"
        ),
        "expected_behavior": (
            "Time = Distance ÷ Speed\n"
            "Time = 180 ÷ 60 = 3 hours\n"
            "Answer: C"
        ),
        "difficulty": "easy",
        "source": "aquarat-speed-distance-time",
    },
    {
        "task_input": (
            "What is 15% of 240?\n"
            "Options:\n"
            "A) 30\n"
            "B) 36\n"
            "C) 40\n"
            "D) 45\n"
            "E) 48"
        ),
        "expected_behavior": (
            "15% of 240 = (15 / 100) × 240\n"
            "= 0.15 × 240\n"
            "= 36\n"
            "Answer: B"
        ),
        "difficulty": "easy",
        "source": "aquarat-percentage",
    },
    {
        "task_input": (
            "A number is increased by 25% and the result is 100. "
            "What is the original number?\n"
            "Options:\n"
            "A) 70\n"
            "B) 75\n"
            "C) 78\n"
            "D) 80\n"
            "E) 85"
        ),
        "expected_behavior": (
            "Let the original number be x.\n"
            "x × 1.25 = 100\n"
            "x = 100 / 1.25\n"
            "x = 80\n"
            "Answer: D"
        ),
        "difficulty": "easy",
        "source": "aquarat-reverse-percentage",
    },
]
