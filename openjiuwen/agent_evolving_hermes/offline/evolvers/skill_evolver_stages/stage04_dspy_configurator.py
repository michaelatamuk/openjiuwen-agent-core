from __future__ import annotations

import logging
from typing import List, Tuple

import dspy

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.dataset_builder import EvalDataset
from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule

_DSPY_LOGGERS = (
    "dspy",
    "dspy.teleprompt",
    "dspy.teleprompt.mipro_optimizer_v2",
    "dspy.teleprompt.bootstrap",
    "dspy.teleprompt.gepa",
    "dspy.predict",
    "dspy.adapters",
)


def configure_dspy_and_prepare_sets(skill_raw: str,
                                    dataset: EvalDataset,
                                    config: EvolverConfig,
                                    console) -> Tuple[SkillModule, List, List]:
    """Configure the DSPy LM, build the baseline module, and convert splits.

    Returns (baseline_module, trainset, valset) ready for GEPA.

    When config.verbose=False (default) DSPy's INFO-level training chatter
    (step banners, bootstrap traces, proposed instruction dumps) is suppressed.
    Set config.verbose=True to restore full DSPy logging.
    """
    console.print("\n[blue]~~~ Evolving Stage 04 - DSPY Configure and Sets Prepare Started ~~~[/blue]")

    log_level = logging.INFO if config.verbose else logging.ERROR
    for name in _DSPY_LOGGERS:
        logging.getLogger(name).setLevel(log_level)

    dspy.configure(lm=dspy.LM(config.optimizer_model))
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")
    baseline_module = SkillModule(skill_raw)

    console.print("[blue]~~~ Evolving Stage 04 - DSPY Configure and Sets Prepare Finished ~~~[/blue]")
    return baseline_module, trainset, valset
