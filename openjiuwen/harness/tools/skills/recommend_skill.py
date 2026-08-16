# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""RecommendSkillTool — offline, matrix-backed skill recommender (Phase 0 only).

Used by SkillUseRail in "recommendation" mode as a drop-in for ListSkillTool.
No LLM call is made; relevance is computed via TF-IDF cosine similarity against
a pre-built scoring matrix (scoring_matrix_*.json files in the oracle dir).

The oracle_dir must be provided explicitly.  When running inside JiuwenSwarm it
is resolved by ``_oracle_dir()`` in config_specs.py, which reads
``react.oracle_dir`` from config.yaml or the ``JIUWENSWARM_ORACLE_DIR`` env var.
There is no default path — if both are absent oracle_dir is None.
Passing ``oracle_dir=None`` means "not configured".

Graceful fallbacks:
  - oracle_dir is None (not configured)        → return all skills, mode="fallback_no_oracle_dir"
  - oracle_dir missing or has no matrix files  → return all skills, mode="fallback_no_matrix"
  - scikit-learn / pandas not installed        → return all skills, mode="fallback_import_error"
  - query produces no matches above threshold  → return all skills, mode="fallback_no_match"
  - any other runtime exception                → return all skills, mode="fallback_error"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from openjiuwen.core.foundation.tool.base import Tool
from openjiuwen.core.single_agent.skills.skill_manager import Skill
from openjiuwen.harness.prompts.tools import build_tool_card
from openjiuwen.harness.tools.base_tool import ToolOutput

logger = logging.getLogger(__name__)


class RecommendSkillTool(Tool):
    """Recommend skills for a query using a pre-computed scoring matrix (Phase 0).

    The recommender is built lazily on first call and cached.  Cache is
    invalidated automatically when the set of active skill names changes.
    """

    def __init__(
        self,
        get_skills: Callable[[], List[Skill]],
        oracle_dir: Optional[Path | str] = None,
        top_k: int = 10,
        sim_threshold: float = 0.25,
        score_threshold: float = 0.20,
        language: str = "cn",
        agent_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            get_skills: Callable returning the current list of active Skill objects.
            oracle_dir: Directory containing ``scoring_matrix_*.json`` files.
                        When None the tool falls back to returning all skills.
                        In JiuwenSwarm this is always resolved by config_specs
                        (defaulting to ~/.jiuwenswarm/agent/workspace/oracle).
            top_k: Maximum number of recommendations to return.
            sim_threshold: Minimum TF-IDF cosine similarity to include a row.
            score_threshold: Minimum weighted score for a recommendation to be kept.
            language: Tool description language ('cn' or 'en').
            agent_id: Optional agent ID for unique tool card ID generation.
        """
        super().__init__(
            build_tool_card("recommend_skill", "RecommendSkillTool", language, agent_id=agent_id)
        )
        self.get_skills = get_skills
        self._oracle_dir: Optional[Path] = (
            Path(oracle_dir).expanduser() if oracle_dir else None
        )
        self._top_k = top_k
        self._sim_threshold = sim_threshold
        self._score_threshold = score_threshold
        self.language = language

        # Cache: keyed by frozenset of skill names present when the recommender was built.
        self._recommender = None
        self._recommender_skill_key: frozenset[str] | None = None

    # ── Tool interface ─────────────────────────────────────────────────────────

    async def invoke(self, inputs: Dict[str, Any], **kwargs) -> ToolOutput:
        """Invoke the recommend_skill tool."""
        oracle_dir_str = str(self._oracle_dir) if self._oracle_dir else None

        query = str(inputs.get("query", "") or "").strip()
        if not query:
            return ToolOutput(
                success=True,
                data={
                    "skills": self._dump_all_skills(),
                    "mode": "fallback_no_query",
                    "oracle_dir": oracle_dir_str,
                },
            )

        # No oracle dir configured — skip recommender entirely.
        if self._oracle_dir is None:
            logger.info(
                "[RecommendSkillTool] oracle_dir not configured; falling back to all skills.",
            )
            return ToolOutput(
                success=True,
                data={
                    "skills": self._dump_all_skills(),
                    "mode": "fallback_no_oracle_dir",
                    "oracle_dir": None,
                },
            )

        # Check if matrix files exist before attempting to load.
        if not self._matrix_exists():
            logger.info(
                "[RecommendSkillTool] No scoring_matrix_*.json found in %s; "
                "falling back to all skills.",
                self._oracle_dir,
            )
            return ToolOutput(
                success=True,
                data={
                    "skills": self._dump_all_skills(),
                    "mode": "fallback_no_matrix",
                    "oracle_dir": oracle_dir_str,
                },
            )

        try:
            recommender = self._get_recommender()
        except ImportError as exc:
            logger.warning(
                "[RecommendSkillTool] Cannot import _recommender package: %s; "
                "falling back to all skills.",
                exc,
            )
            return ToolOutput(
                success=True,
                data={
                    "skills": self._dump_all_skills(),
                    "mode": "fallback_import_error",
                    "oracle_dir": oracle_dir_str,
                },
            )
        except Exception as exc:
            logger.error(
                "[RecommendSkillTool] Failed to build recommender from %s: %s; "
                "falling back to all skills.",
                self._oracle_dir,
                exc,
                exc_info=True,
            )
            return ToolOutput(
                success=True,
                data={
                    "skills": self._dump_all_skills(),
                    "mode": "fallback_error",
                    "oracle_dir": oracle_dir_str,
                },
            )

        try:
            raw_results = recommender.recommend(
                query,
                sim_threshold=self._sim_threshold,
                score_threshold=self._score_threshold,
                top_k=self._top_k,
            )
        except Exception as exc:
            logger.error(
                "[RecommendSkillTool] recommend() raised: %s; falling back to all skills.",
                exc,
                exc_info=True,
            )
            return ToolOutput(
                success=True,
                data={
                    "skills": self._dump_all_skills(),
                    "mode": "fallback_error",
                    "oracle_dir": oracle_dir_str,
                },
            )

        # Filter to skills that actually exist in the current skills dir.
        current_skills_by_name: Dict[str, Skill] = {
            s.name: s for s in (self.get_skills() or [])
        }
        ranked: List[Dict[str, Any]] = []
        seen_skills: set[str] = set()

        for rec in raw_results:
            skill_name = rec.get("skill", "")
            if skill_name not in current_skills_by_name:
                continue
            if skill_name in seen_skills:
                # Multiple metrics can produce duplicate skill entries — keep highest score only.
                continue
            seen_skills.add(skill_name)
            skill = current_skills_by_name[skill_name]
            skill_dict = skill.asdict(include_directory=True)
            skill_dict["skill_md_path"] = str(Path(skill.directory) / "SKILL.md")
            skill_dict["score"] = rec.get("score", 0.0)
            skill_dict["n_examples"] = rec.get("n_examples", 0)
            ranked.append(skill_dict)

        if not ranked:
            logger.debug(
                "[RecommendSkillTool] No matrix skills matched current skill set for query %r; "
                "falling back to all skills.",
                query[:80],
            )
            return ToolOutput(
                success=True,
                data={
                    "skills": self._dump_all_skills(),
                    "mode": "fallback_no_match",
                    "oracle_dir": oracle_dir_str,
                },
            )

        return ToolOutput(
            success=True,
            data={
                "skills": ranked,
                "mode": "recommendation",
                "oracle_dir": oracle_dir_str,
            },
        )

    async def stream(self, inputs: Dict[str, Any], **kwargs) -> AsyncIterator[Any]:
        if False:
            yield None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _matrix_exists(self) -> bool:
        """Return True if the oracle dir is configured, exists, and contains matrix files."""
        if self._oracle_dir is None:
            return False
        if not self._oracle_dir.exists():
            return False
        return any(self._oracle_dir.glob("scoring_matrix_*.json"))

    def _get_recommender(self):
        """Return a cached SkillRecommender, rebuilding if the skill set changed."""
        current_key = frozenset(s.name for s in (self.get_skills() or []))
        if self._recommender is not None and self._recommender_skill_key == current_key:
            return self._recommender

        from openjiuwen.harness.tools.skills._recommender.recommender_builder import (
            build_recommender,
        )

        logger.debug(
            "[RecommendSkillTool] Building recommender from %s (skill set changed or first call).",
            self._oracle_dir,
        )
        self._recommender = build_recommender(self._oracle_dir, variant="baseline", embedder_method="tfidf")
        self._recommender_skill_key = current_key
        return self._recommender

    def _dump_all_skills(self) -> List[Dict[str, Any]]:
        """Dump all current enabled skills as serializable dicts."""
        results: List[Dict[str, Any]] = []
        for skill in self.get_skills() or []:
            skill_dict = skill.asdict(include_directory=True)
            skill_dict["skill_md_path"] = str(Path(skill.directory) / "SKILL.md")
            results.append(skill_dict)
        return results
