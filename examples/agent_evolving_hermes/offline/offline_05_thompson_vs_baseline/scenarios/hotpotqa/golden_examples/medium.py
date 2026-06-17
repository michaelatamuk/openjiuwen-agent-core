# coding: utf-8
# Medium — genuine two-hop bridge questions; the answer cannot be found
# without first resolving an intermediate entity.  The baseline guesses the
# answer directly; the evolved skill names both hops.

GOLDEN_EXAMPLES_MEDIUM = [
    {
        "task_input": (
            "What country did the inventor of the telephone move to "
            "after leaving Scotland?"
        ),
        "expected_behavior": (
            "The telephone was invented by Alexander Graham Bell.\n"
            "Bell was born in Edinburgh, Scotland, and emigrated to Canada "
            "in 1870 before later becoming a US citizen.\n"
            "**Answer: Canada**"
        ),
        "difficulty": "medium",
        "source": "hotpotqa-bridge-inventor",
    },
    {
        "task_input": (
            "In which city is the law school attended by the 44th "
            "President of the United States?"
        ),
        "expected_behavior": (
            "The 44th President of the United States was Barack Obama.\n"
            "Obama attended Harvard Law School, which is located in "
            "Cambridge, Massachusetts.\n"
            "**Answer: Cambridge, Massachusetts**"
        ),
        "difficulty": "medium",
        "source": "hotpotqa-bridge-president-education",
    },
    {
        "task_input": (
            "What ocean does the river that runs through the Amazon rainforest "
            "flow into?"
        ),
        "expected_behavior": (
            "The river that runs through the Amazon rainforest is the Amazon River.\n"
            "The Amazon River flows eastward through South America and empties "
            "into the Atlantic Ocean near Marajó Island in Brazil.\n"
            "**Answer: Atlantic Ocean**"
        ),
        "difficulty": "medium",
        "source": "hotpotqa-bridge-geography-river",
    },
    {
        "task_input": (
            "What programming language was created by the person who invented Python?"
        ),
        "expected_behavior": (
            "Python was created by Guido van Rossum.\n"
            "Before Python, Guido van Rossum created the ABC programming language "
            "at CWI in the Netherlands, which directly influenced Python's design.\n"
            "**Answer: ABC**"
        ),
        "difficulty": "medium",
        "source": "hotpotqa-bridge-programming-language",
    },
]
