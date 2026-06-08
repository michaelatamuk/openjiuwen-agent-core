from typing import Optional

from offline import EvolverConfig


class SkillEvolverParams:
    def __init__(self, skill_name: str, eval_source: str = "synthetic", external_sources: Optional[list] = None,
                 iterations: Optional[int] = None, config: Optional[EvolverConfig] = None, reuse_dataset: bool = False,
                 min_improvement: float = 0.0, prior_baseline_score_single: Optional[float] = None,
                 prior_baseline_score_multi: Optional[float] = None, prior_baseline_dims_multi = None,
                 prebuilt_skill: Optional[dict] = None, prebuilt_dataset = None,
                 prebuilt_baseline_module = None, prebuilt_trainset: Optional[list] = None,
                 prebuilt_valset: Optional[list] = None, console = None):
        self.skill_name: str = skill_name
        self.eval_source: str = eval_source
        self.external_sources: Optional[list] = external_sources
        self.iterations: Optional[int] = iterations
        self.config: Optional[EvolverConfig] = config
        self.reuse_dataset: bool = reuse_dataset
        self.min_improvement: float = min_improvement
        self.prior_baseline_score_single: Optional[float] = prior_baseline_score_single
        self.prior_baseline_score_multi: Optional[float] = prior_baseline_score_multi
        self.prior_baseline_dims_multi = prior_baseline_dims_multi
        self.prebuilt_skill: Optional[dict] = prebuilt_skill
        self.prebuilt_dataset = prebuilt_dataset
        self.prebuilt_baseline_module = prebuilt_baseline_module
        self.prebuilt_trainset: Optional[list] = prebuilt_trainset
        self.prebuilt_valset: Optional[list] = prebuilt_valset
        self.console = console

