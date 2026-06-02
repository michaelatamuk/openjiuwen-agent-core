# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Multi-objective scoring state — dynamic weight tracker.

Persists per-output-dir across GEPA runs in ``mo_state.json``.

Weight update rules (applied after each run):
  - Dimension improved (evolved > baseline):
      stagnation = 0,  weight = max(0.5, weight − 0.10)
  - Dimension did NOT improve:
      stagnation += 1
      if stagnation >= 3:  weight += 0.25
  - Normalize so sum(weights) = n_dims  (= 5)

This means stagnant dimensions get increasing attention while
dimensions that reliably improve are down-weighted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_N_DIMS = 5
_DIM_NAMES = [
    "correctness",
    "procedure_following",
    "conciseness",
    "completeness",
    "specificity",
]
_NO_REGRESSION_THRESHOLD = 0.02


@dataclass
class MultiObjectiveState:
    """Dynamic weight state for multi-objective GEPA scoring."""

    weights: List[float] = field(default_factory=lambda: [1.0] * _N_DIMS)
    stagnation: List[int] = field(default_factory=lambda: [0] * _N_DIMS)

    # ── Aggregate ─────────────────────────────────────────────────────────────

    def aggregate(self, subscores: List[float]) -> float:
        """Weighted mean of sub-scores.  Result is in [0, 1]."""
        total_w = sum(self.weights)
        if total_w == 0:
            return sum(subscores) / _N_DIMS
        return sum(w * s for w, s in zip(self.weights, subscores)) / total_w

    # ── No-regression check ───────────────────────────────────────────────────

    def no_regression_passed(
        self,
        evolved: List[float],
        baseline: List[float],
    ) -> Tuple[bool, List[str]]:
        """Return (all_passed, list_of_failed_dim_names).

        A dimension fails if ``evolved[i] < baseline[i] - threshold``.
        """
        failed: List[str] = []
        for name, e, b in zip(_DIM_NAMES, evolved, baseline):
            if e < b - _NO_REGRESSION_THRESHOLD:
                failed.append(name)
        return len(failed) == 0, failed

    # ── Weight update ─────────────────────────────────────────────────────────

    def update_weights(self, evolved: List[float], baseline: List[float]) -> None:
        """Update stagnation counts and re-normalize weights."""
        for i, (e, b) in enumerate(zip(evolved, baseline)):
            if e > b:
                self.stagnation[i] = 0
                self.weights[i] = max(0.5, self.weights[i] - 0.10)
            else:
                self.stagnation[i] += 1
                if self.stagnation[i] >= 3:
                    self.weights[i] += 0.25

        # Normalize: sum(weights) = n_dims
        total = sum(self.weights)
        if total > 0:
            self.weights = [w * _N_DIMS / total for w in self.weights]

    # ── Persistence ───────────────────────────────────────────────────────────

    @classmethod
    def load_or_create(cls, path: Path) -> "MultiObjectiveState":
        """Load state from *path* or return a fresh default state."""
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                return cls(
                    weights=list(map(float, raw.get("weights", [1.0] * _N_DIMS))),
                    stagnation=list(map(int, raw.get("stagnation", [0] * _N_DIMS))),
                )
            except Exception:
                pass  # corrupt file → start fresh
        return cls()

    def save(self, path: Path) -> None:
        """Atomically write state to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "dim_names": _DIM_NAMES,
                    "weights": [round(w, 6) for w in self.weights],
                    "stagnation": self.stagnation,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.rename(path)

    # ── Human-readable weight display ─────────────────────────────────────────

    def weights_str(self) -> str:
        """Compact weight string for display, e.g. '1.00 1.00 1.25 0.90 0.90'."""
        return "  ".join(f"{w:.2f}" for w in self.weights)
