from __future__ import annotations

from typing import List, Tuple

import dspy

from openjiuwen.agent_evolving_hermes.offline.config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.dataset_builder import EvalDataset
from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule


def configure_dspy_and_prepare_sets(
    skill_raw: str,
    dataset: EvalDataset,
    config: EvolverConfig,
) -> Tuple[SkillModule, List, List]:
    """Configure the DSPy LM, build the baseline module, and convert splits.

    Returns (baseline_module, trainset, valset) ready for GEPA.
    """
    dspy.configure(lm=dspy.LM(config.optimizer_model))
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")
    baseline_module = SkillModule(skill_raw)
    return baseline_module, trainset, valset
