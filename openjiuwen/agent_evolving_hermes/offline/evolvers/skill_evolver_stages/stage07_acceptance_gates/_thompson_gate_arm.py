from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class _Arm:
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
