# coding: utf-8
# Easy — questions that appear to require one hop but whose expected_behavior
# explicitly names the connecting fact.  Baseline gives a bare answer;
# expected output names and links the intermediate fact.

GOLDEN_EXAMPLES_EASY = [
    {
        "task_input": (
            "What is the capital of the country in which the Eiffel Tower is located?"
        ),
        "expected_behavior": (
            "The Eiffel Tower is located in France.\n"
            "The capital of France is Paris.\n"
            "**Answer: Paris**"
        ),
        "difficulty": "easy",
        "source": "hotpotqa-bridge-geography",
    },
    {
        "task_input": (
            "The musical 'West Side Story' is a modern adaptation of which "
            "Shakespeare play?"
        ),
        "expected_behavior": (
            "West Side Story is a modern retelling of Romeo and Juliet.\n"
            "Romeo and Juliet was written by William Shakespeare.\n"
            "**Answer: Romeo and Juliet**"
        ),
        "difficulty": "easy",
        "source": "hotpotqa-bridge-literature",
    },
    {
        "task_input": (
            "In which country is the company that manufactures the iPhone headquartered?"
        ),
        "expected_behavior": (
            "The iPhone is manufactured by Apple Inc.\n"
            "Apple Inc. is headquartered in Cupertino, California, "
            "in the United States.\n"
            "**Answer: United States**"
        ),
        "difficulty": "easy",
        "source": "hotpotqa-bridge-company",
    },
    {
        "task_input": (
            "What sport is played at Wimbledon, the world's oldest Grand Slam tournament?"
        ),
        "expected_behavior": (
            "Wimbledon is the world's oldest Grand Slam tournament.\n"
            "Grand Slam tournaments are part of professional tennis.\n"
            "**Answer: Tennis**"
        ),
        "difficulty": "easy",
        "source": "hotpotqa-bridge-sports",
    },
]
