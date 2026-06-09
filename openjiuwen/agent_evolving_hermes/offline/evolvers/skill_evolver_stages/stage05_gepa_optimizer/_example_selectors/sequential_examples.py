from typing import List


# ── Sequential (legacy) ───────────────────────────────────────────────────────

class SequentialExampleSelector:
    """Return all training examples unchanged — preserves current behaviour.

    ``update()`` is a no-op; no state is persisted.
    """

    def __init__(self, trainset: List) -> None:
        self._trainset = list(trainset)

    def select(self) -> List:
        return list(self._trainset)

    def update(self, examples: List, fitnesses: List[float]) -> None:
        pass  # no-op
