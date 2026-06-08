# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Level 3 Thompson Sampling — Acceptance Gate.

Decides whether to deploy an evolved skill candidate.

┌────────────────────────┬───────────────────────────────────────────────┐
│ ThresholdAcceptanceGate│ Accept if improvement >= min_improvement.     │
│  (legacy)              │ Identical to the original apply_acceptance_   │
│                        │ gate() function.  ts_confidence is None.      │
├────────────────────────┼───────────────────────────────────────────────┤
│ ThompsonAcceptanceGate │ Adds a second gate on top of the hard        │
│  (TS)                  │ threshold: P(θ_candidate > θ_deployed) must  │
│                        │ reach ts_acceptance_confidence.  This        │
│                        │ prevents deploying one-off lucky runs and    │
│                        │ requires sustained evidence of improvement.  │
└────────────────────────┴───────────────────────────────────────────────┘

Both classes implement AcceptanceGateProtocol (from protocols.py).

Factory
-------
    make_acceptance_gate(config, min_improvement) → AcceptanceGateProtocol

The factory reads ``config.ts_acceptance_gate`` to pick the implementation.

Thompson arm state persists per-skill to
``<ts_state_dir>/ts_gate_<skill_name>.json``.
Two arms are maintained per skill:
  ``<skill>__candidate`` — updated every time a candidate is evaluated
  ``<skill>__deployed``  — updated only when a candidate is accepted
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.evolvers.selection.acceptance_gates.base import BaseAcceptanceGate
from openjiuwen.agent_evolving_hermes.offline.evolvers.selection.acceptance_gates.thompson import ThompsonAcceptanceGate
from openjiuwen.agent_evolving_hermes.offline.evolvers.selection.acceptance_gates.threshold import \
    ThresholdAcceptanceGate
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig


# ── Factory ───────────────────────────────────────────────────────────────────

def make_acceptance_gate(
    config: "EvolverConfig",
    min_improvement: float = 0.0,
) -> "ThresholdAcceptanceGate | ThompsonAcceptanceGate":
    """Return the correct acceptance gate based on ``config.ts_acceptance_gate``.

    Parameters
    ----------
    config:
        EvolverConfig instance.  Read fields: ``ts_acceptance_gate``,
        ``ts_acceptance_confidence``, ``ts_acceptance_n_samples``,
        ``ts_state_dir``, ``output_dir``.
    min_improvement:
        The hard minimum improvement threshold.  Applied by both gate types.
        Passed in separately (rather than read from config) to keep the
        existing call-site signature.

    Usage (inside skill_evolver_single)::

        gate = make_acceptance_gate(config, min_improvement)
        accepted, ts_conf = gate.decide(
            improvement, evolved_score, skill_name,
            evolved_text, cross_run_delta, output_dir, console,
        )
    """
    if getattr(config, "ts_acceptance_gate", False):
        return ThompsonAcceptanceGate(config, min_improvement=min_improvement)
    return ThresholdAcceptanceGate(min_improvement=min_improvement)
