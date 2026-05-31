# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 12 hand-crafted examples: 4 easy · 4 medium · 4 hard
#
# Design rationale: easy examples are visible even to a shallow skill.
# Medium and hard examples (security bugs, concurrency, N+1) only surface in
# a genuinely deep review.  TS will learn to focus budget on these hard
# examples, which are most discriminating between evolved variants.
# ══════════════════════════════════════════════════════════════════════════════

example_01 =     {
        "task_input": (
            "Review this Python function:\n\n"
            "def process_items(items=[]):\n"
            "    items.append('new')\n"
            "    return items"
        ),
        "expected_behavior": (
            "Must identify the mutable default argument bug. `items=[]` is "
            "evaluated once at function definition time; every call without "
            "an explicit argument shares the same list across calls. "
            "Fix: use `items=None` and set `if items is None: items = []` "
            "inside the function body."
        ),
        "difficulty": "easy",
        "category": "python-gotchas",
        "source": "golden",
    }

example_02 =     {
        "task_input": (
            "Review this code:\n\n"
            "list = [1, 2, 3]\n"
            "dict = {'a': 1}\n"
            "result = list + [4]"
        ),
        "expected_behavior": (
            "Must flag that `list` and `dict` shadow Python built-ins. "
            "This will cause confusing NameError or TypeError later when "
            "the names are needed as built-in types. Recommend renaming "
            "to domain-specific names such as `items` and `config`."
        ),
        "difficulty": "easy",
        "category": "naming",
        "source": "golden",
    }

example_03 = {
        "task_input": (
            "Review this utility module:\n\n"
            "import os, sys, json, re, pathlib\n\n"
            "def get_config(path):\n"
            "    with open(path) as f:\n"
            "        return json.load(f)"
        ),
        "expected_behavior": (
            "Must identify that `os`, `sys`, `re`, and `pathlib` are imported "
            "but never used. Unused imports increase load time, confuse readers, "
            "and trigger linter warnings. Only `json` is needed here."
        ),
        "difficulty": "easy",
        "category": "style",
        "source": "golden",
    }

example_04 = {
        "task_input": (
            "Review this script:\n\n"
            "def main():\n"
            "    print('Starting application')\n"
            "    run_server()\n\n"
            "main()"
        ),
        "expected_behavior": (
            "Must flag the missing `if __name__ == '__main__': main()` guard. "
            "Calling `main()` directly at module level means it executes on "
            "import, breaking test frameworks and any code that imports this module."
        ),
        "difficulty": "easy",
        "category": "structure",
        "source": "golden",
    }


GOLDEN_EXAMPLES_EASY = [ example_01, example_02, example_03, example_04 ]
