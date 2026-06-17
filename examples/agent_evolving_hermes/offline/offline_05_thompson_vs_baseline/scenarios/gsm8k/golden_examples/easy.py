# coding: utf-8
# Easy — single arithmetic operation; baseline may accidentally match the number
# but lacks the reasoning chain the expected_behavior requires.

GOLDEN_EXAMPLES_EASY = [
    {
        "task_input": (
            "Sarah has 24 cookies. She gives 9 cookies to her classmates. "
            "How many cookies does she have left?"
        ),
        "expected_behavior": (
            "Sarah starts with 24 cookies and gives away 9.\n"
            "24 - 9 = 15\n"
            "**Answer: 15 cookies**"
        ),
        "difficulty": "easy",
        "source": "gsm8k-single-step-subtraction",
    },
    {
        "task_input": (
            "A movie ticket costs $12. How much do 4 tickets cost in total?"
        ),
        "expected_behavior": (
            "Each ticket costs $12 and we need 4 tickets.\n"
            "4 × $12 = $48\n"
            "**Answer: $48**"
        ),
        "difficulty": "easy",
        "source": "gsm8k-single-step-multiplication",
    },
    {
        "task_input": (
            "Tom reads 15 pages every day. How many pages will he read in 6 days?"
        ),
        "expected_behavior": (
            "Tom reads 15 pages per day for 6 days.\n"
            "15 × 6 = 90 pages\n"
            "**Answer: 90 pages**"
        ),
        "difficulty": "easy",
        "source": "gsm8k-single-step-multiplication",
    },
    {
        "task_input": (
            "A bag contains 7 red marbles and 11 blue marbles. "
            "How many marbles are there in total?"
        ),
        "expected_behavior": (
            "The bag has 7 red marbles and 11 blue marbles.\n"
            "7 + 11 = 18 marbles\n"
            "**Answer: 18 marbles**"
        ),
        "difficulty": "easy",
        "source": "gsm8k-single-step-addition",
    },
]
