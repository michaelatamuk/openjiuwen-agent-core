# coding: utf-8
# Hard — multi-step problems involving rates, percentages, or compound reasoning.
# The baseline gives a single number; the expected_behavior requires every
# intermediate step to be written out explicitly.

GOLDEN_EXAMPLES_HARD = [
    {
        "task_input": (
            "A car travels 180 miles in 3 hours, then 160 miles in 2 more hours. "
            "What is the car's average speed for the entire journey?"
        ),
        "expected_behavior": (
            "Total distance: 180 + 160 = 340 miles\n"
            "Total time: 3 + 2 = 5 hours\n"
            "Average speed: 340 ÷ 5 = 68 mph\n"
            "**Answer: 68 mph**"
        ),
        "difficulty": "hard",
        "source": "gsm8k-multi-step-average-speed",
    },
    {
        "task_input": (
            "A shop owner buys a watch for $80, marks it up by 25%, "
            "then offers a 10% discount off the marked price. "
            "What is the final selling price?"
        ),
        "expected_behavior": (
            "Price after 25% markup: $80 × 1.25 = $100\n"
            "Price after 10% discount: $100 × (1 − 0.10) = $100 × 0.90 = $90\n"
            "**Answer: $90**"
        ),
        "difficulty": "hard",
        "source": "gsm8k-multi-step-markup-discount",
    },
    {
        "task_input": (
            "In a school of 500 students, 60% are girls. "
            "Of the girls, 30% play a sport. "
            "How many girls play a sport?"
        ),
        "expected_behavior": (
            "Number of girls: 60% × 500 = 0.60 × 500 = 300\n"
            "Girls who play a sport: 30% × 300 = 0.30 × 300 = 90\n"
            "**Answer: 90 girls**"
        ),
        "difficulty": "hard",
        "source": "gsm8k-multi-step-nested-percentage",
    },
    {
        "task_input": (
            "A water tank holds 1,200 litres when full. A pipe fills it at "
            "40 litres per minute, but a leak drains it at 10 litres per minute. "
            "How many minutes does it take to fill the empty tank?"
        ),
        "expected_behavior": (
            "Net fill rate: 40 − 10 = 30 litres per minute\n"
            "Time to fill: 1,200 ÷ 30 = 40 minutes\n"
            "**Answer: 40 minutes**"
        ),
        "difficulty": "hard",
        "source": "gsm8k-multi-step-net-rate",
    },
]
