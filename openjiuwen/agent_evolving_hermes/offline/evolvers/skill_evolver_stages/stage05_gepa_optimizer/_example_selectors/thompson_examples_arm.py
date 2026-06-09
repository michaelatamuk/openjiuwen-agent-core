import random
from dataclasses import dataclass


@dataclass
class _ExampleArm:
    key: str
    alpha: float = 1.0   # prior + high-fitness hits
    beta: float = 1.0    # prior + low-fitness misses

    def update(self, fitness: float, threshold: float = 0.3) -> None:
        # Threshold lowered from 0.5 to 0.3: partial progress (the agent getting
        # roughly a third of the expected keywords right) counts as a success.
        # With threshold=0.5, examples where the skill is misaligned but improving
        # accumulate β and get deprioritised before GEPA has a chance to fix the
        # skill.  0.3 keeps partially-productive examples in rotation.
        if fitness >= threshold:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)