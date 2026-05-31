# coding: utf-8
"""Example 5 — Thompson Sampling vs baseline GEPA: a side-by-side comparison.

Goal
----
Show concretely what Thompson Sampling adds to GEPA by running the *same*
skill through two independent evolution passes — one with plain GEPA, one
with all TS levels active — and printing a three-way comparison:

    BASELINE  ──►  GEPA (no TS)  ──►  GEPA (with TS)

Domain: Python code review
Why this domain?  Code review quality has clear gradations:
  • surface level   — "check for bugs, ensure readability"
  • medium depth    — SQL injection, off-by-one, bare except
  • expert level    — TOCTOU race, N+1 query, thread-safety, path traversal

The baseline skill is deliberately shallow.  A naive GEPA run tends to
over-optimise for the easy examples (they dominate the training set).
TS (Level 2) learns which examples are most *discriminating* and focuses
GEPA's limited budget there, producing a deeper evolved skill.
TS (Level 3) then prevents deploying a lucky one-off improvement — it
requires P(candidate > deployed) ≥ 0.75 across 100 Monte Carlo draws.

Both runs use an identical 12-example golden dataset so the comparison is
fair.

Model
-----
Uses DeepSeek-Chat (https://api.deepseek.com) for both the optimiser LLM
and the LLM-as-judge evaluator.  Set your key:

    export DEEPSEEK_API_KEY=sk-...

Usage
-----
    python examples/agent_evolving_hermes/offline/offline_05_thompson_vs_baseline.py

Or with a custom key inline:
    DEEPSEEK_API_KEY=sk-... python offline_05_thompson_vs_baseline.py
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Optional

from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, evolve_single_skill

# ── Model ──────────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-PLACEHOLDER")
DEEPSEEK_MODEL   = "deepseek/deepseek-chat"
DEEPSEEK_BASE    = "https://api.deepseek.com"

# ── Run settings ───────────────────────────────────────────────────────────────
SKILL_NAME      = "code-review"
ITERATIONS      = 5          # GEPA iterations per run
TS_BATCH_SIZE   = 4          # TS example selector: pick 4 of 6 train examples

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE SKILL — deliberately shallow so there is room to improve
# ══════════════════════════════════════════════════════════════════════════════

SKILL_FRONTMATTER = """\
name: code-review
description: Reviews Python code and provides feedback on bugs, style, and improvements
"""

SKILL_BODY = """\
# Python Code Review

## Instructions

When asked to review Python code, look for the following:

- Check for obvious bugs in the code
- Ensure the code is readable and follows good practices
- Look for potential performance issues
- Suggest improvements where needed

Provide your feedback in a clear format and list the issues you find.
Be constructive and helpful.
"""

# ══════════════════════════════════════════════════════════════════════════════
# GOLDEN DATASET — 12 hand-crafted examples: 4 easy · 4 medium · 4 hard
#
# Design rationale: easy examples are visible even to a shallow skill.
# Medium and hard examples (security bugs, concurrency, N+1) only surface in
# a genuinely deep review.  TS will learn to focus budget on these hard
# examples, which are most discriminating between evolved variants.
# ══════════════════════════════════════════════════════════════════════════════

GOLDEN_EXAMPLES = [
    # ── EASY ──────────────────────────────────────────────────────────────────
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
    },
    # ── MEDIUM ────────────────────────────────────────────────────────────────
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
    },
    # ── HARD ──────────────────────────────────────────────────────────────────
    {
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
    },
    {
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
    },
    {
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
    },
    {
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
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# Helpers — setup
# ══════════════════════════════════════════════════════════════════════════════

def _write_skill(skills_root: Path, name: str, frontmatter: str, body: str) -> Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(f"---\n{frontmatter.strip()}\n---\n{body}")
    return skill_path


def _write_golden_dataset(skills_root: Path, name: str, examples: list) -> Path:
    """Write examples to the golden_dataset path that eval_source='golden' reads."""
    out = skills_root / name / "golden_dataset"
    out.mkdir(parents=True, exist_ok=True)

    shuffled = list(examples)
    random.shuffle(shuffled)
    n          = len(shuffled)
    train_end  = int(n * 0.50)
    val_end    = train_end + int(n * 0.25)
    splits     = {"train": shuffled[:train_end],
                  "val":   shuffled[train_end:val_end],
                  "holdout": shuffled[val_end:]}

    for split_name, rows in splits.items():
        path = out / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    return out


def _read_latest_evolved(output_dir: Path, skill_name: str) -> Optional[str]:
    candidates = sorted((output_dir / skill_name).glob("*/evolved_skill.md"))
    if not candidates:
        # Accept regression too, to always show something
        candidates = sorted((output_dir / skill_name).glob("*/evolved_REGRESSION.md"))
    return candidates[-1].read_text() if candidates else None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers — display
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title: str, width: int = 70) -> None:
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def _print_skill(label: str, text: str, max_lines: int = 35) -> None:
    _banner(label)
    lines = text.strip().splitlines()
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"  … ({len(lines) - max_lines} more lines not shown)")


def _print_metrics(metrics: dict) -> None:
    bs = metrics.get("baseline_score", 0.0)
    es = metrics.get("evolved_score",  0.0)
    delta = es - bs
    sign  = "+" if delta >= 0 else ""
    print(f"    Baseline score  : {bs:.4f}")
    print(f"    Evolved score   : {es:.4f}")
    print(f"    Improvement     : {sign}{delta:.4f}")
    print(f"    Elapsed         : {metrics.get('elapsed_seconds', 0.0):.1f}s")
    print(f"    Accepted        : {'✓ Yes' if metrics.get('accepted') else '✗ No (saved as REGRESSION)'}")
    xd = metrics.get("cross_run_delta")
    if xd is not None:
        print(f"    vs prior best   : {'+' if xd>=0 else ''}{xd:.4f}")
    checks = metrics.get("constraint_checks", [])
    if checks:
        print("    Constraints:")
        for check in checks:
            icon = "✓" if check["passed"] else "✗"
            print(f"      {icon} {check['name']:20s}  {check['message']}")


def _print_ts_insights(ts_state_dir: Path, skill_name: str) -> None:
    """Read TS arm state files and print a human-readable summary."""
    examples_path = ts_state_dir / f"ts_examples_{skill_name}.json"
    gate_path     = ts_state_dir / f"ts_gate_{skill_name}.json"

    if examples_path.exists():
        with open(examples_path) as fh:
            state = json.load(fh)
        arms = state.get("arms", {})
        if arms:
            print("\n  TS Level 2 — Example Selector (arm state after training)")
            print("  Higher α/(α+β) = example was more discriminating, received more budget")
            ranked = sorted(
                [(k, v.get("alpha", 1.0), v.get("beta", 1.0)) for k, v in arms.items()],
                key=lambda x: x[1] / (x[1] + x[2]),
                reverse=True,
            )
            for rank, (key, alpha, beta) in enumerate(ranked, 1):
                ratio = alpha / (alpha + beta)
                bar   = "█" * int(ratio * 16) + "░" * (16 - int(ratio * 16))
                print(f"    #{rank}  {bar}  α={alpha:.1f} β={beta:.1f}  "
                      f"p={ratio:.2f}  key={key[:16]}…")

    if gate_path.exists():
        with open(gate_path) as fh:
            state = json.load(fh)
        cand = state.get("candidate", {})
        dep  = state.get("deployed",  {})
        if cand:
            ac, bc = cand.get("alpha", 1.0), cand.get("beta", 1.0)
            ad, bd = dep.get("alpha",  1.0), dep.get("beta", 1.0)
            print(f"\n  TS Level 3 — Acceptance Gate")
            print(f"    Candidate arm : α={ac:.1f}, β={bc:.1f}  "
                  f"(mean={ac/(ac+bc):.2f})")
            print(f"    Deployed  arm : α={ad:.1f}, β={bd:.1f}  "
                  f"(mean={ad/(ad+bd):.2f})")


# ══════════════════════════════════════════════════════════════════════════════
# Main demo
# ══════════════════════════════════════════════════════════════════════════════

def run_demo() -> None:
    os.environ.setdefault("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    os.environ.setdefault("DEEPSEEK_API_BASE", DEEPSEEK_BASE)

    workdir      = Path(tempfile.mkdtemp(prefix="gepa_ts_demo_"))
    skills_root  = workdir / "skills"
    output_no_ts = workdir / "output_no_ts"
    output_ts    = workdir / "output_ts"
    ts_state_dir = workdir / "ts_state"
    ts_state_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWorking directory : {workdir}")
    print(f"Model             : {DEEPSEEK_MODEL}  ({DEEPSEEK_BASE})")
    print(f"GEPA iterations   : {ITERATIONS}")
    print(f"Golden examples   : {len(GOLDEN_EXAMPLES)}  "
          f"({sum(1 for e in GOLDEN_EXAMPLES if e['difficulty']=='easy')} easy / "
          f"{sum(1 for e in GOLDEN_EXAMPLES if e['difficulty']=='medium')} medium / "
          f"{sum(1 for e in GOLDEN_EXAMPLES if e['difficulty']=='hard')} hard)")

    # ── Step 1: Write baseline skill + golden dataset ─────────────────────────
    skill_path = _write_skill(skills_root, SKILL_NAME, SKILL_FRONTMATTER, SKILL_BODY)
    golden_dir = _write_golden_dataset(skills_root, SKILL_NAME, GOLDEN_EXAMPLES)
    print(f"\nBaseline skill    : {skill_path}")
    print(f"Golden dataset    : {golden_dir}")

    baseline_text = skill_path.read_text()
    _print_skill("① BASELINE SKILL — before any evolution", baseline_text)

    # ── Step 2: GEPA run without Thompson Sampling ────────────────────────────
    _banner("② GEPA — without Thompson Sampling")
    print("  Example selector : all training examples, equal weight")
    print("  Acceptance gate  : threshold only (improvement ≥ 0.0)")
    print()

    config_no_ts = EvolverConfig(
        skills_root  = skills_root,
        output_dir   = output_no_ts,
        iterations   = ITERATIONS,
        optimizer_model = DEEPSEEK_MODEL,
        eval_model      = DEEPSEEK_MODEL,
        # Thompson Sampling — all OFF
        ts_skill_scheduler   = False,
        ts_example_selector  = False,
        ts_acceptance_gate   = False,
    )
    metrics_no_ts = evolve_single_skill(
        SKILL_NAME, "golden", config=config_no_ts, min_improvement=0.0
    )
    _print_metrics(metrics_no_ts)

    evolved_no_ts = _read_latest_evolved(output_no_ts, SKILL_NAME)
    _print_skill("  Evolved skill (no TS)", evolved_no_ts or "[not produced]")

    # ── Step 3: Restore baseline skill ───────────────────────────────────────
    _write_skill(skills_root, SKILL_NAME, SKILL_FRONTMATTER, SKILL_BODY)

    # ── Step 4: GEPA run WITH Thompson Sampling (Level 2 + Level 3) ──────────
    _banner("③ GEPA — with Thompson Sampling (Level 2 + Level 3)")
    print(f"  Level 2 — Example Selector  : selects top {TS_BATCH_SIZE} of "
          f"{int(len(GOLDEN_EXAMPLES)*0.5)} train examples per iteration")
    print("    TS learns which examples best distinguish good vs bad evolved skills")
    print("    → hard examples (security, concurrency) accumulate higher α")
    print()
    print(f"  Level 3 — Acceptance Gate   : P(candidate > deployed) ≥ 0.75")
    print("    Monte Carlo (100 draws) — prevents accepting a lucky one-off run")
    print()

    config_ts = EvolverConfig(
        skills_root  = skills_root,
        output_dir   = output_ts,
        iterations   = ITERATIONS,
        optimizer_model = DEEPSEEK_MODEL,
        eval_model      = DEEPSEEK_MODEL,
        # Thompson Sampling — Level 2 + Level 3 ON
        ts_skill_scheduler   = False,          # L1 only matters for --all runs
        ts_example_selector  = True,
        ts_example_batch_size = TS_BATCH_SIZE,
        ts_acceptance_gate   = True,
        ts_acceptance_confidence = 0.75,
        ts_acceptance_n_samples  = 100,
        ts_state_dir = ts_state_dir,
    )
    metrics_ts = evolve_single_skill(
        SKILL_NAME, "golden", config=config_ts, min_improvement=0.0
    )
    _print_metrics(metrics_ts)
    _print_ts_insights(ts_state_dir, SKILL_NAME)

    evolved_ts = _read_latest_evolved(output_ts, SKILL_NAME)
    _print_skill("  Evolved skill (with TS)", evolved_ts or "[not produced]")

    # ── Step 5: Three-way comparison table ───────────────────────────────────
    _banner("COMPARISON — Baseline  ·  GEPA no-TS  ·  GEPA with-TS")

    bs   = metrics_no_ts.get("baseline_score", 0.0)
    s_no = metrics_no_ts.get("evolved_score",  0.0)
    s_ts = metrics_ts.get("evolved_score",     0.0)
    W = 16

    print(f"\n  {'':32s}  {'Baseline':>{W}}  {'No-TS':>{W}}  {'With-TS':>{W}}")
    print(f"  {'─'*32}  {'─'*W}  {'─'*W}  {'─'*W}")
    print(f"  {'Holdout score':32s}  {bs:>{W}.4f}  {s_no:>{W}.4f}  {s_ts:>{W}.4f}")
    d_no = s_no - bs
    d_ts = s_ts - bs
    print(f"  {'Δ over baseline':32s}  {'—':>{W}}  "
          f"{('+' if d_no>=0 else '') + f'{d_no:.4f}':>{W}}  "
          f"{('+' if d_ts>=0 else '') + f'{d_ts:.4f}':>{W}}")
    accepted_no = '✓ yes' if metrics_no_ts.get('accepted') else '✗ no'
    accepted_ts = '✓ yes' if metrics_ts.get('accepted')    else '✗ no'
    print(f"  {'Accepted':32s}  {'—':>{W}}  {accepted_no:>{W}}  {accepted_ts:>{W}}")
    print(f"  {'Acceptance gate':32s}  {'—':>{W}}  {'threshold':>{W}}  {'TS confidence':>{W}}")
    print(f"  {'Examples / iteration':32s}  {'—':>{W}}  {'all train':>{W}}  "
          f"{f'top {TS_BATCH_SIZE} (TS-ranked)':>{W}}")
    print(f"  {'Hard examples targeted':32s}  {'—':>{W}}  {'no':>{W}}  {'yes (learned)':>{W}}")

    winner = ("With-TS" if s_ts > s_no
              else "No-TS"  if s_no > s_ts
              else "Tie")
    print(f"\n  ▶  Best evolution: {winner}  "
          f"(Δ = {s_ts - s_no:+.4f} in favour of TS run)")

    # ── Step 6: Where to look ─────────────────────────────────────────────────
    print(f"\n  Output files:")
    print(f"    No-TS run   →  {output_no_ts}/{SKILL_NAME}/")
    print(f"    With-TS run →  {output_ts}/{SKILL_NAME}/")
    print(f"    TS state    →  {ts_state_dir}/")
    print()
    print("  To inspect TS arm state directly:")
    print(f"    python -m json.tool {ts_state_dir}/ts_examples_{SKILL_NAME}.json")
    print(f"    python -m json.tool {ts_state_dir}/ts_gate_{SKILL_NAME}.json")
    print()
    print("  To re-run with your own skill, edit SKILL_FRONTMATTER / SKILL_BODY")
    print("  and add your own entries to GOLDEN_EXAMPLES at the top of this file.")


if __name__ == "__main__":
    run_demo()
