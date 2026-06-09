from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from ._base_gate import BaseAcceptanceGate
from ._thompson_gate_arm import _Arm


# ── Thompson Sampling gate ────────────────────────────────────────────────────

class ThompsonAcceptanceGate(BaseAcceptanceGate):
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

    def __init__(self, config: EvolverConfig, min_improvement: float = 0.0) -> None:
        self._min = min_improvement
        self._confidence = float(getattr(config, "ts_acceptance_confidence", 0.75))
        self._n_samples = int(getattr(config, "ts_acceptance_n_samples", 100))
        raw_state_dir = getattr(config, "ts_state_dir", None)
        self._state_dir: Path = (Path(raw_state_dir) if raw_state_dir else Path(config.output_dir))
        self._state_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-skill arm persistence ─────────────────────────────────────────────

    def _state_path(self, skill_name: str) -> Path:
        return self._state_dir / f"ts_gate_{skill_name}.json"

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

        def _arm(key: str) -> _Arm:
            d = raw.get(key, {})
            return _Arm(name=key,
                        alpha=float(d.get("alpha", 1.0)),
                        beta=float(d.get("beta", 1.0)),
                        n_runs=int(d.get("n_runs", 0)))

        return _arm(candidate_key), _arm(deployed_key)

    def _save_arms(self, skill_name: str, candidate: _Arm, deployed: _Arm) -> None:
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

    def _ts_confidence(self, candidate: _Arm, deployed: _Arm) -> float:
        """Estimate P(θ_candidate > θ_deployed) via Monte Carlo."""
        wins = sum(candidate.sample() > deployed.sample()
                   for _ in range(self._n_samples))
        return wins / self._n_samples

    # ── Main decision ─────────────────────────────────────────────────────────

    def decide(self, improvement: float, evolved_score: float, skill_name: str, evolved_text: str,
               cross_run_delta: Optional[float], output_dir: Path, console) -> Tuple[bool, Optional[float]]:
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
        sign = "+" if improvement >= 0 else ""
        conf_pct = int(round(ts_conf * 100))
        need_pct = int(round(self._confidence * 100))

        console.print("\nAcceptance gate  (TS-AcceptanceGate):")

        # Check 1: score change
        if hard_pass:
            console.print(f"  Check 1 — score change: {sign}{improvement:.4f}"
                          f"  [green]✓ meets minimum {self._min:.4f}[/green]")
        else:
            console.print(f"  Check 1 — score change: {sign}{improvement:.4f}"
                          f"  [red]✗ below minimum {self._min:.4f}[/red]")

        # Check 2: TS confidence (always shown so user sees the number)
        conf_label = (f"{conf_pct}% of {self._n_samples} draws: "
                      f"candidate > deployed skill")
        if ts_pass:
            console.print(f"  Check 2 — {conf_label}"
                          f"  [green]✓ meets {need_pct}%[/green]")
        elif hard_pass:
            # Score passed but confidence failed — highlight in red
            console.print(f"  Check 2 — {conf_label}"
                          f"  [red]✗ need ≥{need_pct}%[/red]")
        else:
            # Score already failed — show confidence in dim (informational only)
            console.print(f"  Check 2 — {conf_label}"
                          f"  [dim]✗ need ≥{need_pct}%[/dim]")

        # Final decision
        if accepted:
            console.print("  Decision: [green]ACCEPTED — evolved skill will be deployed[/green]")
        else:
            if not hard_pass:
                console.print("  Decision: [red]REJECTED — score did not improve[/red]")
            else:
                console.print("  Decision: [red]REJECTED — improvement may be luck,"
                              " not enough evidence yet[/red]")
            (output_dir / "evolved_REGRESSION.md").write_text(evolved_text, encoding="utf-8")
            console.print("  [dim](saved to evolved_REGRESSION.md)[/dim]")
            trend = self._trend_line(cross_run_delta)
            if trend:
                color = "green" if cross_run_delta >= 0 else "red"
                console.print(f"  [{color}]{trend.strip()}[/{color}]")

        return accepted, ts_conf
