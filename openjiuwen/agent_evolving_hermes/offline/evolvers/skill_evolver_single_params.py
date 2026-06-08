from typing import Optional

from .skill_evolver_config import EvolverConfig


class SkillEvolverParams:
    def __init__(self,
                 skill_name: str,
                 eval_source: str = "synthetic",
                 external_sources: Optional[list] = None,
                 reuse_dataset: bool = False,
                 min_improvement: float = 0.0,
                 prior_baseline_score_single: Optional[float] = None,
                 prior_baseline_score_multi: Optional[float] = None,
                 prior_baseline_dims_multi=None,
                 prior_metrics=None,
                 cached_path=None,
                 *,
                 config: EvolverConfig,
                 console,
                 prebuilt_skill: dict,
                 prebuilt_dataset,
                 prebuilt_baseline_module,
                 prebuilt_trainset: list,
                 prebuilt_valset: list):
        self.skill_name: str = skill_name
        self.eval_source: str = eval_source
        self.external_sources: Optional[list] = external_sources
        self.config: EvolverConfig = config
        self.reuse_dataset: bool = reuse_dataset
        self.min_improvement: float = min_improvement
        self.prior_baseline_score_single: Optional[float] = prior_baseline_score_single
        self.prior_baseline_score_multi: Optional[float] = prior_baseline_score_multi
        self.prior_baseline_dims_multi = prior_baseline_dims_multi
        self.prior_metrics = prior_metrics
        self.cached_path = cached_path
        self.prebuilt_skill: dict = prebuilt_skill
        self.prebuilt_dataset = prebuilt_dataset
        self.prebuilt_baseline_module = prebuilt_baseline_module
        self.prebuilt_trainset: list = prebuilt_trainset
        self.prebuilt_valset: list = prebuilt_valset
        self.console = console
