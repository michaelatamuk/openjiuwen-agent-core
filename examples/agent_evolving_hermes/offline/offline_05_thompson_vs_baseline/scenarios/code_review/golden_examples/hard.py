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
            "Review this file upload handler:\n\n"
            "import os\n\n"
            "def save_upload(filename, data):\n"
            "    path = f'/uploads/{filename}'\n"
            "    if not os.path.exists(path):\n"
            "        with open(path, 'wb') as f:\n"
            "            f.write(data)\n"
            "    else:\n"
            "        raise FileExistsError(f'{path} already exists')"
        ),
        "expected_behavior": (
            "Must identify two issues: (1) TOCTOU race condition — between "
            "`os.path.exists()` and `open()` another process can create the file, "
            "causing data loss or a security issue. Atomic fix: `open(path, 'xb')` "
            "which raises FileExistsError atomically. (2) Path traversal — a "
            "filename like `../../etc/passwd` escapes the uploads directory. "
            "Must validate that the resolved path stays inside `/uploads/`."
        ),
        "difficulty": "hard",
        "category": "security",
        "source": "golden",
    }

example_02 = {
        "task_input": (
            "Review this Django view:\n\n"
            "def get_order_details(request, order_id):\n"
            "    order = Order.objects.get(id=order_id)\n"
            "    items = []\n"
            "    for line in order.lineitems.all():\n"
            "        product = Product.objects.get(id=line.product_id)\n"
            "        items.append({'name': product.name, 'qty': line.quantity})\n"
            "    return JsonResponse({'order': order.id, 'items': items})"
        ),
        "expected_behavior": (
            "Must identify the N+1 query problem. For an order with N line items "
            "this executes 1 query for the order plus N queries for products — "
            "O(N) database round trips. Fix: "
            "`order.lineitems.select_related('product').all()` fetches everything "
            "in 1-2 queries. This is a critical performance issue that becomes "
            "catastrophic at scale."
        ),
        "difficulty": "hard",
        "category": "performance",
        "source": "golden",
    }

example_03 = {
        "task_input": (
            "Review this singleton counter:\n\n"
            "class Counter:\n"
            "    _instance = None\n\n"
            "    @classmethod\n"
            "    def get_instance(cls):\n"
            "        if cls._instance is None:\n"
            "            cls._instance = cls()\n"
            "        return cls._instance\n\n"
            "    def __init__(self):\n"
            "        self.count = 0\n\n"
            "    def increment(self):\n"
            "        self.count += 1"
        ),
        "expected_behavior": (
            "Must identify two concurrency issues: (1) The singleton check in "
            "`get_instance()` is not thread-safe — two threads can both see "
            "`_instance is None` simultaneously and create duplicate instances. "
            "Fix: guard with a threading.Lock. (2) `self.count += 1` is not "
            "atomic under contention. Fix: use threading.Lock around the increment "
            "or use threading.local for per-thread counters."
        ),
        "difficulty": "hard",
        "category": "concurrency",
        "source": "golden",
    }

example_04 = {
        "task_input": (
            "Review this event pipeline:\n\n"
            "class EventProcessor:\n"
            "    def __init__(self):\n"
            "        self.handlers = []\n"
            "        self.processed = []\n\n"
            "    def register(self, handler):\n"
            "        self.handlers.append(handler)\n\n"
            "    def process(self, event):\n"
            "        result = handler(event) for handler in self.handlers\n"
            "        self.processed.append((event, list(result)))\n"
            "        return result"
        ),
        "expected_behavior": (
            "Must identify three issues: (1) Syntax error — the generator "
            "expression on the `result =` line is invalid Python; needs list() "
            "or explicit parentheses. (2) Memory leak — `self.processed` grows "
            "unbounded; every processed event is kept forever, exhausting memory "
            "in high-volume pipelines. Use a bounded deque or periodic flush. "
            "(3) No error handling — if any handler raises, the event is silently "
            "lost and subsequent handlers are skipped."
        ),
        "difficulty": "hard",
        "category": "correctness",
        "source": "golden",
    }

example_05 = {
    "task_input": (
        "Review this deserializer:\n\n"
        "import pickle\n\n"
        "def load_session(cookie_value: bytes):\n"
        "    return pickle.loads(cookie_value)"
    ),
    "expected_behavior": (
        "Must identify the critical insecure deserialization vulnerability. "
        "`pickle.loads` on untrusted data (a cookie) allows arbitrary code "
        "execution — an attacker can craft a cookie whose `__reduce__` method "
        "runs any OS command. Fix: never deserialize untrusted input with "
        "`pickle`; use `json.loads` or a signed/encrypted token like `itsdangerous.URLSafeTimedSerializer`. "
        "This is an OWASP Top 10 vulnerability (A08 — Software and Data Integrity Failures)."
    ),
    "difficulty": "hard",
    "category": "security",
    "source": "golden",
}

example_06 = {
    "task_input": (
        "Review this password storage function:\n\n"
        "import hashlib\n\n"
        "def store_password(password: str) -> str:\n"
        "    return hashlib.md5(password.encode()).hexdigest()"
    ),
    "expected_behavior": (
        "Must identify the weak cryptography issue. MD5 is a fast hash — an "
        "attacker with the hash can crack common passwords in seconds using "
        "rainbow tables or GPU brute-force. Two further issues: (1) no salt, "
        "so identical passwords produce identical hashes exposing all users "
        "at once. (2) MD5 is not a password-hashing function. "
        "Fix: use `bcrypt`, `argon2`, or `hashlib.pbkdf2_hmac` with a random "
        "`salt` and a work factor that makes brute-forcing expensive."
    ),
    "difficulty": "hard",
    "category": "security",
    "source": "golden",
}

example_07 = {
    "task_input": (
        "Review this cache implementation:\n\n"
        "class Cache:\n"
        "    _store = {}\n\n"
        "    def get(self, key):\n"
        "        return self._store.get(key)\n\n"
        "    def set(self, key, value):\n"
        "        self._store[key] = value"
    ),
    "expected_behavior": (
        "Must identify two issues: (1) `_store` is a class-level attribute — "
        "it is shared across ALL instances of Cache. Creating two Cache objects "
        "gives them the same underlying dict; data written through one is "
        "visible through the other. Fix: initialise `self._store = {}` in "
        "`__init__`. (2) No eviction policy or `maxsize` — the dict grows "
        "unbounded, causing a memory leak under sustained load. "
        "Consider `functools.lru_cache` or set an explicit max size."
    ),
    "difficulty": "hard",
    "category": "correctness",
    "source": "golden",
}

example_08 = {
    "task_input": (
        "Review this lock usage:\n\n"
        "import threading\n\n"
        "_lock = threading.Lock()\n\n"
        "def update_counter(counter, delta):\n"
        "    _lock.acquire()\n"
        "    counter['value'] += delta\n"
        "    _lock.release()"
    ),
    "expected_behavior": (
        "Must flag that the lock is released in the happy path only. "
        "If `counter['value'] += delta` raises a KeyError or any other "
        "exception, `_lock.release()` is never called and every subsequent "
        "thread that tries to acquire the lock deadlocks forever. "
        "Fix: use the lock as a context manager — `with _lock:` — which "
        "guarantees release via `__exit__` even on exceptions. "
        "Alternatively wrap in `try/finally` with `_lock.release()` in the "
        "`finally` block. This is the canonical use of `threading.Lock`."
    ),
    "difficulty": "hard",
    "category": "concurrency",
    "source": "golden",
}

GOLDEN_EXAMPLES_HARD = [
    example_01, example_02, example_03, example_04,
    example_05, example_06, example_07, example_08,
]

