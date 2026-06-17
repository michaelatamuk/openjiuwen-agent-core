# coding: utf-8
# Hard — comparison questions that require resolving two entities and comparing
# a specific attribute.  The baseline picks one answer without comparing;
# the evolved skill identifies both entities, extracts the attribute for each,
# and explicitly states which one satisfies the comparison.

GOLDEN_EXAMPLES_HARD = [
    {
        "task_input": (
            "Which was founded earlier: Harvard University or the United States "
            "of America?"
        ),
        "expected_behavior": (
            "Harvard University was founded in 1636 in Cambridge, Massachusetts.\n"
            "The United States of America was founded in 1776 with the "
            "Declaration of Independence.\n"
            "Since 1636 comes before 1776, Harvard University was founded earlier.\n"
            "**Answer: Harvard University**"
        ),
        "difficulty": "hard",
        "source": "hotpotqa-comparison-founding-dates",
    },
    {
        "task_input": (
            "Is the Amazon River longer or shorter than the Nile River?"
        ),
        "expected_behavior": (
            "The Amazon River is approximately 6,400 km (3,976 miles) long.\n"
            "The Nile River is approximately 6,650 km (4,130 miles) long.\n"
            "Since 6,400 < 6,650, the Amazon is shorter than the Nile.\n"
            "**Answer: Shorter than the Nile**"
        ),
        "difficulty": "hard",
        "source": "hotpotqa-comparison-river-length",
    },
    {
        "task_input": (
            "Did Marie Curie or Albert Einstein win their first Nobel Prize earlier?"
        ),
        "expected_behavior": (
            "Marie Curie won her first Nobel Prize in Physics in 1903 "
            "(shared with Pierre Curie and Henri Becquerel).\n"
            "Albert Einstein won his Nobel Prize in Physics in 1921.\n"
            "Since 1903 comes before 1921, Marie Curie won her first Nobel Prize earlier.\n"
            "**Answer: Marie Curie**"
        ),
        "difficulty": "hard",
        "source": "hotpotqa-comparison-nobel-prizes",
    },
    {
        "task_input": (
            "Which country has a larger population: Brazil or Russia?"
        ),
        "expected_behavior": (
            "Brazil has a population of approximately 215 million people "
            "(as of the most recent census data).\n"
            "Russia has a population of approximately 144 million people.\n"
            "Since 215 million > 144 million, Brazil has a larger population.\n"
            "**Answer: Brazil**"
        ),
        "difficulty": "hard",
        "source": "hotpotqa-comparison-population",
    },
]
