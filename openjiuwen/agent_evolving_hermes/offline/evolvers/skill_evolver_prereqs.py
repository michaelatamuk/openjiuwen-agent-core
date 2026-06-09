# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pre-evolution prerequisite builder.

Runs stages 1–4 of the single-skill evolution pipeline exactly once and
returns an :class:`EvolutionPrereqs` container.

Call :func:`build_evolution_prereqs` before :func:`evolve_single_skill`
so the evolver receives fully initialized objects rather than building
them internally.  This mirrors what ``step_01_build_skill_dataset_and_dspy``
does in the demo pipeline, but for production / CLI / batch callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from ._console_maker import _make_console
from .skill_evolver_stages.stage01_skill_finder_and_loader.skill_finder_and_loader import find_and_load_skill
from .skill_evolver_stages.stage02_skill_constraint_validator.skill_constraint_validator import validate_skill_constraints
from .skill_evolver_stages.stage03_dataset_builder.dataset_builder import build_or_load_dataset
from .skill_evolver_stages.stage04_dspy_configurator.dspy_configurator import configure_dspy_and_prepare_sets
from .skill_evolver_config import EvolverConfig


@dataclass
class EvolutionPrereqs:
    """Objects produced by stages 1–4, ready to pass into ``evolve_single_skill``.

    Attributes
    ----------
    skill:
        Skill dict returned by ``find_and_load_skill``
        (keys: ``"raw"``, ``"name"``, ``"frontmatter"``).
    dataset:
        :class:`EvalDataset` with train / val / holdout splits.
    baseline_module:
        :class:`SkillModule` wrapping the baseline skill text.
    trainset:
        DSPy-formatted training examples.
    valset:
        DSPy-formatted validation examples.
    prior_metrics:
        Metrics from the most recent previous run loaded by
        ``find_and_load_skill``, or ``None`` when no prior run exists.
        Used for cross-run delta in stage 8 / stage 11.
    cached_path:
        Filesystem path where the dataset was cached by stage 3,
        or ``None`` when the dataset was generated fresh and not cached.
    console:
        Rich console created (or forwarded) by this function.
    """

    skill: dict
    dataset: Any
    baseline_module: Any
    trainset: List[Any]
    valset: List[Any]
    prior_metrics: Any
    cached_path: Any
    console: Any


def build_evolution_prereqs(
    skill_name: str,
    config: EvolverConfig,
    eval_source: str,
    external_sources: Optional[list] = None,
    reuse_dataset: bool = False,
    console=None,
) -> EvolutionPrereqs:
    """Run stages 1–4 and return all objects needed by ``evolve_single_skill``.

    Parameters
    ----------
    skill_name:
        Skill identifier (sub-directory name under ``config.skills_root``).
    config:
        Fully populated :class:`EvolverConfig`.  ``config.output_dir`` is
        used as the dataset cache root.
    eval_source:
        ``"synthetic"`` | ``"external"`` | ``"golden"``
    external_sources:
        External log sources — only relevant when ``eval_source="external"``.
    reuse_dataset:
        When ``True``, reuse the most recently cached dataset for this skill
        instead of regenerating it.
    console:
        Existing Rich :class:`Console` to write to.  A fresh console is
        created when ``None``.

    Returns
    -------
    EvolutionPrereqs
        All stage 1–4 artefacts packaged for forwarding to
        :class:`~.skill_evolver_single_params.SkillEvolverParams`.
    """
    if console is None:
        console = _make_console()

    dataset_output_dir = Path(config.output_dir) / skill_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1 — find and load skill
    skill, prior_metrics = find_and_load_skill(skill_name, config, console)

    # Stage 2 — validate baseline constraints
    validate_skill_constraints(skill["raw"], config, console, stage_label="Baseline")

    # Stage 3 — build / load eval dataset
    dataset, cached_path = build_or_load_dataset(
        skill_name, skill["raw"], eval_source, external_sources,
        config, dataset_output_dir, reuse_dataset, console,
    )

    # Stage 4 — configure DSPy LM and prepare train/val splits
    baseline_module, trainset, valset = configure_dspy_and_prepare_sets(
        skill["raw"], dataset, config, console
    )

    return EvolutionPrereqs(
        skill=skill,
        dataset=dataset,
        baseline_module=baseline_module,
        trainset=trainset,
        valset=valset,
        prior_metrics=prior_metrics,
        cached_path=cached_path,
        console=console,
    )
