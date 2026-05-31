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
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from openjiuwen.agent_evolving_hermes.offline.config import EvolverConfig


# ── Shared Beta arm ───────────────────────────────────────────────────────────

@dataclass
class _BetaArm:
    name: str
    alpha: float = 1.0
    beta: float = 1.0
    n_runs: int = 0

    def update(self, improvement: float) -> None:
        self.n_runs += 1
        if improvement > 0.0:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)


# ── Threshold gate (legacy) ───────────────────────────────────────────────────

class ThresholdAcceptanceGate:
    """Accept if improvement >= min_improvement — identical to the original.

    ``decide()`` always returns ``(accepted, None)``; the second element is
    always None because there is no TS confidence to report.
    """

    def __init__(self, min_improvement: float = 0.0) -> None:
        self._min = min_improvement

    def decide(
        self,
        improvement: float,
        evolved_score: float,
        skill_name: str,
        evolved_text: str,
        cross_run_delta: Optional[float],
        output_dir: Path,
        console,
    ) -> Tuple[bool, Optional[float]]:
        accepted = improvement >= self._min

        if not accepted:
            (output_dir / "evolved_REGRESSION.md").write_text(evolved_text)
            console.print(
                f"[yellow]⚠ Improvement {improvement:+.4f} < threshold "
                f"{self._min:+.4f} — not deploying "
                f"(saved to evolved_REGRESSION.md)[/yellow]"
            )
            if cross_run_delta is not None:
                color = "green" if cross_run_delta >= 0 else "red"
                console.print(
                    f"[{color}]Cross-run delta vs prior evolved: "
                    f"{cross_run_delta:+.4f}[/{color}]"
                )
        else:
            if improvement < 0:
                console.print(
                    f"[yellow]⚠ Improvement {improvement:+.4f} is negative "
                    f"(accepted because threshold is {self._min:+.4f})[/yellow]"
                )

        return accepted, None


# ── Thompson Sampling gate ────────────────────────────────────────────────────

class ThompsonAcceptanceGate:
    """Accept only when both the hard threshold AND a TS confidence test pass.

    Two Beta(α, β) arms are maintained per skill:

    candidate arm  — updated on every evaluation run
    deployed arm   — updated only when a candidate is accepted and deployed

    Acceptance requires BOTH:
      1. ``improvement >= min_improvement``              (hard gate)
      2. ``P(θ_candidate > θ_deployed) >= confidence``  (Thompson gate)

    The TS gate prevents deploying a one-off lucky run.  The confidence
    requirement means the candidate must reliably outperform the deployed
    skill across Monte Carlo draws before it is accepted.

    Arm state persists to ``<state_dir>/ts_gate_<skill_name>.json``.
    """

    _STATE_PREFIX = "ts_gate_"

    def __init__(
        self,
        config: "EvolverConfig",
        min_improvement: float = 0.0,
    ) -> None:
        self._min = min_improvement
        self._confidence = float(getattr(config, "ts_acceptance_confidence", 0.75))
        self._n_samples = int(getattr(config, "ts_acceptance_n_samples", 100))
        raw_state_dir = getattr(config, "ts_state_dir", None)
        self._state_dir: Path = (
            Path(raw_state_dir) if raw_state_dir else Path(config.output_dir)
        )
        self._state_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-skill arm persistence ─────────────────────────────────────────────

    def _state_path(self, skill_name: str) -> Path:
        return self._state_dir / f"{self._STATE_PREFIX}{skill_name}.json"

    def _load_arms(self, skill_name: str) -> Tuple[_BetaArm, _BetaArm]:
        candidate_key = f"{skill_name}__candidate"
        deployed_key = f"{skill_name}__deployed"
        path = self._state_path(skill_name)
        raw: dict = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text())
            except Exception:
                pass

        def _arm(key: str) -> _BetaArm:
            d = raw.get(key, {})
            return _BetaArm(
                name=key,
                alpha=float(d.get("alpha", 1.0)),
                beta=float(d.get("beta", 1.0)),
                n_runs=int(d.get("n_runs", 0)),
            )

        return _arm(candidate_key), _arm(deployed_key)

    def _save_arms(
        self,
        skill_name: str,
        candidate: _BetaArm,
        deployed: _BetaArm,
    ) -> None:
        path = self._state_path(skill_name)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    candidate.name: {
                        "alpha": candidate.alpha,
                        "beta": candidate.beta,
                        "n_runs": candidate.n_runs,
                    },
                    deployed.name: {
                        "alpha": deployed.alpha,
                        "beta": deployed.beta,
                        "n_runs": deployed.n_runs,
                    },
                },
                indent=2,
            )
        )
        tmp.rename(path)

    # ── TS confidence ─────────────────────────────────────────────────────────

    def _ts_confidence(self, candidate: _BetaArm, deployed: _BetaArm) -> float:
        """Estimate P(θ_candidate > θ_deployed) via Monte Carlo."""
        wins = sum(
            candidate.sample() > deployed.sample()
            for _ in range(self._n_samples)
        )
        return wins / self._n_samples

    # ── Main decision ─────────────────────────────────────────────────────────

    def decide(
        self,
        improvement: float,
        evolved_score: float,
        skill_name: str,
        evolved_text: str,
        cross_run_delta: Optional[float],
        output_dir: Path,
        console,
    ) -> Tuple[bool, Optional[float]]:
        candidate_arm, deployed_arm = self._load_arms(skill_name)

        # Always record this run into the candidate arm
        candidate_arm.update(improvement)

        hard_pass = improvement >= self._min
        ts_conf = self._ts_confidence(candidate_arm, deployed_arm)
        ts_pass = ts_conf >= self._confidence
        accepted = hard_pass and ts_pass

        # Update deployed arm only on acceptance
        if accepted:
            deployed_arm.update(improvement)

        self._save_arms(skill_name, candidate_arm, deployed_arm)

        # ── Console output ────────────────────────────────────────────────────
        if not hard_pass:
            (output_dir / "evolved_REGRESSION.md").write_text(evolved_text)
            console.print(
                f"[yellow]⚠ Hard gate failed: improvement {improvement:+.4f} < "
                f"threshold {self._min:+.4f} "
                f"(saved to evolved_REGRESSION.md)[/yellow]"
            )
        elif not ts_pass:
            (output_dir / "evolved_REGRESSION.md").write_text(evolved_text)
            console.print(
                f"[yellow]⚠ Thompson gate failed: confidence "
                f"{ts_conf:.0%} < {self._confidence:.0%} required "
                f"(saved to evolved_REGRESSION.md)[/yellow]"
            )
            console.print(
                f"[dim]Hard improvement {improvement:+.4f} passed, but not enough "
                f"evidence this candidate beats the deployed skill.[/dim]"
            )
        else:
            if improvement < 0:
                console.print(
                    f"[yellow]⚠ Improvement {improvement:+.4f} negative but threshold "
                    f"allows it; TS confidence {ts_conf:.0%} ≥ "
                    f"{self._confidence:.0%}[/yellow]"
                )
            else:
                console.print(
                    f"[green]✓ TS confidence {ts_conf:.0%} ≥ "
                    f"{self._confidence:.0%} — Thompson gate passed[/green]"
                )

        if not accepted and cross_run_delta is not None:
            color = "green" if cross_run_delta >= 0 else "red"
            console.print(
                f"[{color}]Cross-run delta vs prior evolved: "
                f"{cross_run_delta:+.4f}[/{color}]"
            )

        return accepted, ts_conf


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
