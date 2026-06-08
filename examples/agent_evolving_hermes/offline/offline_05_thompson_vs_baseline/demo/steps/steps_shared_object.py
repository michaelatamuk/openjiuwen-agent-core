from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class SharedEvolutionObjects:
    """Objects built once and reused by every GEPA training pass.

    Attributes
    ----------
    skill:
        Skill dict returned by ``find_and_load_skill`` (keys include
        ``"raw"``, ``"name"``, ``"frontmatter"``).
    dataset:
        :class:`EvalDataset` instance with train / val / holdout splits.
    baseline_module:
        :class:`SkillModule` wrapping the baseline skill text, ready for
        DSPy optimisation.  DSPy optimisers work on copies of the module,
        so the same instance is safe to pass to multiple sequential passes.
    trainset:
        DSPy-formatted training examples (list of ``dspy.Example``).
    valset:
        DSPy-formatted validation examples.
    """

    skill: dict
    dataset: Any
    baseline_module: Any
    trainset: List[Any]
    valset: List[Any]
