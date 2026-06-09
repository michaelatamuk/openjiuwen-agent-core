import json
from pathlib import Path
from typing import List, Dict

from .thompson_examples_arm import _ExampleArm


# ── Thompson Sampling example selector ────────────────────────────────────────

class ThompsonExampleSelector:
    """Select training examples via Thompson Sampling.

    On each ``select()`` call a Beta sample is drawn for every example and
    the top-*batch_size* examples are returned.  After GEPA completes,
    ``update()`` adjusts the Beta arms based on the per-example fitness
    scores produced by the inner ``skill_fitness_metric``.

    Over repeated runs on the same skill the selector learns which examples
    are most discriminating and concentrates the GEPA budget on them.

    Arm state persists to ``<state_dir>/ts_examples_<skill_name>.json``.
    """

    _STATE_PREFIX = "ts_examples_"

    def __init__(self, trainset: List, skill_name: str, batch_size: int, state_dir: Path) -> None:
        self._trainset = list(trainset)
        self._skill_name = skill_name
        # 0 or larger-than-set means use the full set
        self._batch_size = (
            min(batch_size, len(trainset)) if 0 < batch_size < len(trainset)
            else len(trainset)
        )
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._arms: Dict[str, _ExampleArm] = {}
        self._load()

        # Ensure every current example has an arm (new examples get Beta(1,1))
        for ex in self._trainset:
            k = self._example_key(ex)
            self._arms.setdefault(k, _ExampleArm(key=k))

    # ── Persistence ──────────────────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self._state_dir / f"{self._STATE_PREFIX}{self._skill_name}.json"

    def _load(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for key, d in data.items():
                self._arms[key] = _ExampleArm(key=key,
                                              alpha=float(d.get("alpha", 1.0)),
                                              beta=float(d.get("beta", 1.0)))
        except Exception:
            pass

    def _save(self) -> None:
        path = self._state_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({key: {"alpha": arm.alpha, "beta": arm.beta}
                                   for key, arm in self._arms.items()},
                                  indent=2,))
        tmp.rename(path)

    # ── Public interface ──────────────────────────────────────────────────────

    def select(self) -> List:
        """Return the top-*batch_size* examples ranked by Beta sample."""
        ranked = sorted(
            self._trainset,
            key=lambda ex: self._arms.get(self._example_key(ex),
                                          _ExampleArm(key="")).sample(),
            reverse=True,
        )
        return ranked[: self._batch_size]

    def update(self, examples: List, fitnesses: List[float]) -> None:
        """Adjust Beta arms based on per-example fitness scores.

        *examples* and *fitnesses* correspond to the list returned by the
        most recent ``select()`` call (same order, same length).
        """
        for ex, fitness in zip(examples, fitnesses):
            key = self._example_key(ex)
            arm = self._arms.setdefault(key, _ExampleArm(key=key))
            arm.update(fitness)
        self._save()

    # ── Per-example Beta arm ──────────────────────────────────────────────────────

    @staticmethod
    def _example_key(example) -> str:
        """Short deterministic key derived from an example's task_input."""
        text = getattr(example, "task_input", None) or str(example)
        # Use a stable 10-digit unsigned hash so keys remain short in JSON
        return str(abs(hash(text)) % (10 ** 10))
