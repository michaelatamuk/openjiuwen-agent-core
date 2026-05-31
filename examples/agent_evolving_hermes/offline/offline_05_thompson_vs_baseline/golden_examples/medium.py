# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 12 hand-crafted examples: 4 easy · 4 medium · 4 hard
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


GOLDEN_EXAMPLES_MEDIUM = [ example_01, example_02, example_03, example_04 ]
