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
from pathlib import Path

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from ._base_gate import BaseAcceptanceGate
from ._thompson_gate import ThompsonAcceptanceGate
from ._threshold_gate import ThresholdAcceptanceGate


def apply_acceptance_gates(constraints_passed: bool,
                           config: EvolverConfig,
                           min_improvement: float,
                           improvement: float,
                           evolved_score: float,
                           skill_name: str,
                           evolved_text: str,
                           cross_run_delta: float | None,
                           output_dir: Path,
                           console) -> tuple[bool, float | None]:
    """
    Evaluates whether an evolved skill should be accepted based on configurable
    acceptance gates (Threshold vs. Thompson Sampling).

    Parameters
    ----------
    constraints_passed :
        Whether the skill passed hard constraints (e.g., no-regression checks).
    config :
        Configuration instance determining the gate type and parameters.
    min_improvement :
        The hard minimum improvement threshold.
    improvement :
        The calculated improvement of the evolved skill over the baseline.
    evolved_score :
        The absolute score achieved by the evolved skill.
    skill_name :
        Identifier for the skill being evaluated.
    evolved_text :
        The actual evolved instruction string.
    cross_run_delta :
        The performance difference vs. previous evolution runs, if applicable.
    output_dir :
        Directory for saving gate state.
    console :
        Rich console instance for logging.

    Returns
    -------
    tuple[bool, float | None] :
        (accepted, ts_confidence) — whether the skill was accepted and the
        confidence level (if using Thompson Sampling, otherwise None).
    """
    console.print("[blue]~~~ Evolving Stage 07 - Apply Acceptance Gates Started ~~~[/blue]")

    if not constraints_passed:
        console.print("[yellow]Constraints failed: Skipping acceptance gate.[/yellow]")
        console.print("[blue]~~~ Evolving Stage 07 - Apply Acceptance Gates Finished ~~~[/blue]")
        return False, None

    # Resolve and decide based on config
    use_ts = getattr(config, "ts_acceptance_gate", False)

    gate: BaseAcceptanceGate
    if use_ts:
        gate = ThompsonAcceptanceGate(config, min_improvement=min_improvement)
    else:
        gate = ThresholdAcceptanceGate(min_improvement=min_improvement)

    accepted, ts_conf = gate.decide(improvement, evolved_score, skill_name, evolved_text,cross_run_delta, output_dir,
                                    console)

    console.print(f"[blue]~~~ Evolving Stage 07 - Apply Acceptance Gates Finished (Accepted: {accepted}) ~~~[/blue]")
    return accepted, ts_conf
