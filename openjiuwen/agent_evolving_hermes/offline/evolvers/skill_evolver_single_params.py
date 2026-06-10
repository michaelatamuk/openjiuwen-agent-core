from typing import Optional

from .skill_evolver_config import EvolverConfig


class SkillEvolverParams:
    def __init__(self,
                 skill_name: str,
                 eval_source: str = "synthetic",
                 external_sources: Optional[list] = None,
                 reuse_dataset: bool = False,
                 min_improvement: float = 0.0,
                 prior_baseline_score_holistic: Optional[float] = None,
                 prior_baseline_score_rubrics: Optional[float] = None,
                 prior_baseline_score_graph: Optional[float] = None,
                 prior_baseline_score_checklist: Optional[float] = None,
                 prior_baseline_score_instruction_following: Optional[float] = None,
                 prior_baseline_score_consistency: Optional[float] = None,
                 prior_baseline_dims_rubrics=None,
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
        self.prior_baseline_score_holistic: Optional[float] = prior_baseline_score_holistic
        self.prior_baseline_score_rubrics: Optional[float] = prior_baseline_score_rubrics
        self.prior_baseline_score_graph: Optional[float] = prior_baseline_score_graph
        self.prior_baseline_score_checklist: Optional[float] = prior_baseline_score_checklist
        self.prior_baseline_score_instruction_following: Optional[float] = prior_baseline_score_instruction_following
        self.prior_baseline_score_consistency: Optional[float] = prior_baseline_score_consistency
        self.prior_baseline_dims_rubrics = prior_baseline_dims_rubrics
        self.prior_metrics = prior_metrics
        self.cached_path = cached_path
        self.prebuilt_skill: dict = prebuilt_skill
        self.prebuilt_dataset = prebuilt_dataset
        self.prebuilt_baseline_module = prebuilt_baseline_module
        self.prebuilt_trainset: list = prebuilt_trainset
        self.prebuilt_valset: list = prebuilt_valset
        self.console = console
