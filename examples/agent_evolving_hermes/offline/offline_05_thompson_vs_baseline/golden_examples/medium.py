# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 20 hand-crafted examples: 4 easy · 8 medium · 8 hard
#
# Design rationale: easy examples are visible even to a shallow skill.
# Medium and hard examples (security bugs, concurrency, N+1) only surface in
# a genuinely deep review.  TS will learn to focus budget on these hard
# examples, which are most discriminating between evolved variants.
# ══════════════════════════════════════════════════════════════════════════════


example_01 = {
        "task_input": (
            "Review this database function:\n\n"
            "def get_user(username):\n"
            "    query = \"SELECT * FROM users WHERE name = '\" + username + \"'\"\n"
            "    return db.execute(query)"
        ),
        "expected_behavior": (
            "Must identify the SQL injection vulnerability. String concatenation "
            "inserts user input directly into the query. A crafted username like "
            "``' OR '1'='1`` bypasses authentication. Fix: parameterized queries — "
            "`db.execute('SELECT * FROM users WHERE name = ?', (username,))`."
        ),
        "difficulty": "medium",
        "category": "security",
        "source": "golden",
    }

example_02 = {
        "task_input": (
            "Review this pagination helper:\n\n"
            "def get_page(items, page_num, page_size=10):\n"
            "    start = page_num * page_size\n"
            "    end   = start + page_size\n"
            "    return items[start:end]"
        ),
        "expected_behavior": (
            "Must catch the off-by-one error. With page_num=1 (first page) "
            "start=10, skipping the first ten items. For 1-indexed pages the "
            "formula should be `(page_num - 1) * page_size`. Review must also "
            "note there is no bounds checking — page_num=0 or a large value "
            "silently returns an empty list or wrong slice."
        ),
        "difficulty": "medium",
        "category": "logic",
        "source": "golden",
    }

example_03 = {
        "task_input": (
            "Review this config loader:\n\n"
            "def read_config(path):\n"
            "    try:\n"
            "        f = open(path)\n"
            "        data = json.load(f)\n"
            "        return data\n"
            "    except Exception:\n"
            "        return {}"
        ),
        "expected_behavior": (
            "Must flag two issues: (1) File handle `f` is never closed — "
            "use `with open(path) as f:` instead. (2) Bare `except Exception` "
            "swallows all errors silently including PermissionError, "
            "IsADirectoryError, and JSONDecodeError, making it impossible for "
            "callers to distinguish a missing file from malformed JSON."
        ),
        "difficulty": "medium",
        "category": "resource-management",
        "source": "golden",
    }

example_04 = {
        "task_input": (
            "Review this HTTP retry wrapper:\n\n"
            "def call_api(url, retries=3):\n"
            "    for attempt in range(retries):\n"
            "        try:\n"
            "            resp = requests.get(url, timeout=5)\n"
            "            if resp.status_code == 200:\n"
            "                return resp.json()\n"
            "        except requests.RequestException:\n"
            "            pass\n"
            "    return None"
        ),
        "expected_behavior": (
            "Must identify: (1) No backoff — hammering the server on each retry "
            "can worsen outages; use exponential backoff. (2) Returns None on all "
            "failures, so callers cannot distinguish 'unreachable' from 'bad status'. "
            "(3) Only retries on RequestException, not on non-200 status codes — "
            "a 503 response is not retried at all."
        ),
        "difficulty": "medium",
        "category": "reliability",
        "source": "golden",
    }


example_05 = {
    "task_input": (
        "Review this shell helper:\n\n"
        "import subprocess\n\n"
        "def run_command(user_input: str) -> str:\n"
        "    result = subprocess.run(\n"
        "        f'echo {user_input}',\n"
        "        shell=True, capture_output=True, text=True\n"
        "    )\n"
        "    return result.stdout"
    ),
    "expected_behavior": (
        "Must identify the command injection vulnerability. The f-string "
        "inserts `user_input` directly into a shell command with `shell=True`. "
        "A value like `; rm -rf /` executes as a second command. "
        "Fix: pass arguments as a list without `shell=True` — "
        "`subprocess.run(['echo', user_input], capture_output=True, text=True)`. "
        "If `shell=True` is truly required, sanitise with `shlex.quote(user_input)`. "
        "This is a critical OS command injection risk."
    ),
    "difficulty": "medium",
    "category": "security",
    "source": "golden",
}

example_06 = {
    "task_input": (
        "Review this class definition:\n\n"
        "class UserProfile:\n"
        "    tags = []\n"
        "    settings = {}\n\n"
        "    def add_tag(self, tag):\n"
        "        self.tags.append(tag)\n\n"
        "    def set_setting(self, key, value):\n"
        "        self.settings[key] = value"
    ),
    "expected_behavior": (
        "Must identify that `tags` and `settings` are class-level mutable "
        "attributes shared across ALL instances — adding a tag for user A "
        "also adds it for user B. Fix: initialise them in `__init__`: "
        "`self.tags = []` and `self.settings = {}`. "
        "This is a classic Python gotcha; the fix must be in `__init__`, "
        "not at class level."
    ),
    "difficulty": "medium",
    "category": "python-gotchas",
    "source": "golden",
}

example_07 = {
    "task_input": (
        "Review this generator consumer:\n\n"
        "def process_data(source):\n"
        "    gen = (x * 2 for x in source)\n"
        "    first_pass  = list(gen)\n"
        "    second_pass = list(gen)\n"
        "    return first_pass, second_pass"
    ),
    "expected_behavior": (
        "Must flag that generators can only be iterated once. "
        "After `first_pass = list(gen)` the generator is exhausted; "
        "`second_pass = list(gen)` silently returns an empty list. "
        "Fix: materialise once — `data = list(x * 2 for x in source)` — "
        "and reuse `data`, or recreate the generator for each pass. "
        "No exception is raised; the bug produces silent wrong results."
    ),
    "difficulty": "medium",
    "category": "python-gotchas",
    "source": "golden",
}

example_08 = {
    "task_input": (
        "Review this float comparison:\n\n"
        "def is_tax_rate_valid(rate: float) -> bool:\n"
        "    return rate == 0.1 or rate == 0.2 or rate == 0.3\n\n"
        "def apply_discount(price: float, discount: float) -> float:\n"
        "    if price - discount == 0:\n"
        "        return 0\n"
        "    return price - discount"
    ),
    "expected_behavior": (
        "Must flag the floating-point equality comparisons. Due to IEEE 754, "
        "`0.1 + 0.2 != 0.3` in Python. Direct `==` on floats is unreliable. "
        "Fix: use `math.isclose(rate, 0.1)` with an appropriate `rel_tol` "
        "or `abs_tol`. For currency/tax calculations use `decimal.Decimal` "
        "to avoid floating-point errors entirely. "
        "The `price - discount == 0` check has the same problem."
    ),
    "difficulty": "medium",
    "category": "correctness",
    "source": "golden",
}

GOLDEN_EXAMPLES_MEDIUM = [
    example_01, example_02, example_03, example_04,
    example_05, example_06, example_07, example_08,
]
