# coding: utf-8
"""Pre-step between baseline evaluation and GEPA training passes.

Runs stages 1 / 2 / 3 / 4 of ``evolve_single_skill`` exactly **once** and
returns a :class:`SharedEvolutionObjects` container that all subsequent
GEPA passes (steps 02–05) can reuse.

Why this step exists
--------------------
Each call to ``evolve_single_skill`` would otherwise repeat:

  * Stage 1 — ``find_and_load_skill``       (file I/O)
  * Stage 2 — ``validate_baseline_constraints``  (fast, but noisy when repeated)
  * Stage 3 — ``build_or_load_dataset``     (file I/O + possible LLM calls
                                              for synthetic datasets)
  * Stage 4 — ``configure_dspy_and_prepare_sets``
              (DSPy LM init + dataset splitting)

All four objects are *identical* across every training pass because the
skill text is always restored to the same baseline before each run, and
the golden dataset never changes.

The step is placed here (not inside ``trainings.py``) to keep the
orchestration responsibilities separated and to make it easy to call
independently in tests or ad-hoc scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from rich.console import Console

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage01_skill_finder_and_loader import (
    find_and_load_skill,
)
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage02_baseline_constraint_validator import (
    validate_baseline_constraints,
)
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage03_dataset_builder import (
    build_or_load_dataset,
)
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage04_dspy_configurator import (
    configure_dspy_and_prepare_sets,
)


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


def run_step(
    skills_root: Path,
    skill_name: str,
    model: str,
    output_dir: Path,
    verbose: bool = False,
) -> SharedEvolutionObjects:
    """Build shared evolution objects (stages 1 / 2 / 3 / 4) for reuse by all GEPA passes.

    Parameters
    ----------
    skills_root:
        Root directory that contains the skill sub-directory.
    skill_name:
        Skill identifier (sub-directory name under *skills_root*).
    model:
        DSPy model string — must match what the GEPA passes will use.
    output_dir:
        Directory used by the dataset builder for caching.  Reuse
        ``params.output_baseline`` so that the dataset cache written by
        step_01 is available here without a second write.
    verbose:
        ``True`` → show DSPy / Rich INFO logs.

    Returns
    -------
    SharedEvolutionObjects
        Populated with ``skill``, ``dataset``, ``baseline_module``,
        ``trainset``, ``valset``.
    """
    _banner("① SHARED PREP — stages 1–4 (built once for all GEPA passes)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console = Console(force_terminal=True, width=200)

    evolver_config = EvolverConfig(
        skills_root=skills_root,
        output_dir=output_dir,
        optimizer_model=model,
        eval_model=model,
        verbose=verbose,
    )

    # Stage 1 — find and load skill from disk
    skill, _ = find_and_load_skill(skill_name, evolver_config, console)

    # Stage 2 — validate baseline constraints (run once here; skipped in evolve_single_skill when prebuilt_skill is provided)
    validate_baseline_constraints(skill["raw"], evolver_config, console)

    # Stage 3 — build / load eval dataset
    dataset, _ = build_or_load_dataset(
        skill_name=skill_name,
        skill_raw=skill["raw"],
        eval_source="golden",
        external_sources=None,
        config=evolver_config,
        output_dir=output_dir,
        reuse_dataset=False,
        console=console,
    )

    # Stage 4 — configure DSPy LM and create baseline module + splits
    baseline_module, trainset, valset = configure_dspy_and_prepare_sets(
        skill["raw"], dataset, evolver_config, console
    )

    print(
        f"  Shared objects ready — skill: {len(skill['raw'])} chars | "
        f"dataset: {len(dataset.train or []) + len(dataset.val or []) + len(dataset.holdout or [])} examples | "
        f"train: {len(trainset)} / val: {len(valset)}"
    )

    return SharedEvolutionObjects(
        skill=skill,
        dataset=dataset,
        baseline_module=baseline_module,
        trainset=trainset,
        valset=valset,
    )
