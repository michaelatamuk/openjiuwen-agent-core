# coding: utf-8
# Medium — two-step algebra or ratio/mixture problems; estimation deviates more.

GOLDEN_EXAMPLES_MEDIUM = [
    {
        "task_input": (
            "Two pipes A and B can fill a tank in 12 hours and 18 hours respectively. "
            "If both pipes are opened together, how many hours will it take to fill "
            "the tank?\n"
            "Options:\n"
            "A) 6.0 hours\n"
            "B) 7.0 hours\n"
            "C) 7.2 hours\n"
            "D) 8.0 hours\n"
            "E) 9.0 hours"
        ),
        "expected_behavior": (
            "Rate of pipe A = 1/12 tank per hour\n"
            "Rate of pipe B = 1/18 tank per hour\n"
            "Combined rate = 1/12 + 1/18 = 3/36 + 2/36 = 5/36 tank per hour\n"
            "Time = 1 ÷ (5/36) = 36/5 = 7.2 hours\n"
            "Answer: C"
        ),
        "difficulty": "medium",
        "source": "aquarat-work-rate",
    },
    {
        "task_input": (
            "A solution contains 20% alcohol. How many litres of water must be added "
            "to 50 litres of this solution to make it 10% alcohol?\n"
            "Options:\n"
            "A) 25 litres\n"
            "B) 30 litres\n"
            "C) 40 litres\n"
            "D) 50 litres\n"
            "E) 60 litres"
        ),
        "expected_behavior": (
            "Alcohol in original solution = 20% × 50 = 10 litres\n"
            "Let x litres of water be added.\n"
            "New concentration: 10 / (50 + x) = 10%\n"
            "10 = 0.10 × (50 + x)\n"
            "100 = 50 + x\n"
            "x = 50 litres\n"
            "Answer: D"
        ),
        "difficulty": "medium",
        "source": "aquarat-mixture",
    },
    {
        "task_input": (
            "A man invests $5,000 at simple interest of 8% per annum. "
            "After how many years will the investment amount to $7,000?\n"
            "Options:\n"
            "A) 3 years\n"
            "B) 4 years\n"
            "C) 5 years\n"
            "D) 6 years\n"
            "E) 8 years"
        ),
        "expected_behavior": (
            "Simple interest needed = $7,000 − $5,000 = $2,000\n"
            "SI = P × R × T / 100\n"
            "2,000 = 5,000 × 8 × T / 100\n"
            "2,000 = 400 × T\n"
            "T = 2,000 / 400 = 5 years\n"
            "Answer: C"
        ),
        "difficulty": "medium",
        "source": "aquarat-simple-interest",
    },
    {
        "task_input": (
            "In a class of 60 students the ratio of boys to girls is 2 : 3. "
            "How many girls are in the class?\n"
            "Options:\n"
            "A) 20\n"
            "B) 24\n"
            "C) 30\n"
            "D) 36\n"
            "E) 40"
        ),
        "expected_behavior": (
            "Total parts = 2 + 3 = 5\n"
            "Each part = 60 / 5 = 12 students\n"
            "Girls = 3 parts = 3 × 12 = 36\n"
            "Answer: D"
        ),
        "difficulty": "medium",
        "source": "aquarat-ratio",
    },
]
