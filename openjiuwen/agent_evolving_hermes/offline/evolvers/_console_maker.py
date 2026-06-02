# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Main GEPA evolution orchestration.

Mirrors hermes-agent-self-evolution evolve_skill.py.

New vs original plan:
  - Rich console output (Console, Panel, Table) — matches Hermes exactly
  - Dataset caching: if a dataset already exists for this skill, reuse it
    instead of regenerating (saves LLM cost on repeated runs)
  - Cross-run metrics history: each run appends to metrics_history.jsonl
    so regressions across runs are detectable
  - min_improvement acceptance gate: if improvement < threshold, the evolved
    skill is saved as evolved_REGRESSION.md and a warning is printed, but
    it is NOT written to evolved_skill.md (avoids regressing active skills)
"""
try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def _make_console() -> "Console":
    if _RICH:
        return Console(force_terminal=True, width=200)

    class _FallbackConsole:
        def print(self, *args, **kwargs):
            import re
            text = " ".join(str(a) for a in args)
            text = re.sub(r"\[/?[^\]]*\]", "", text)
            print(text)

        def rule(self, *a, **kw):
            print("-" * 60)

    return _FallbackConsole()  # type: ignore[return-value]
