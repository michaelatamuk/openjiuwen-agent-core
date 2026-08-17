# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""RecommendSkillTool — offline, matrix-backed skill recommender (pure Python).

Used by SkillUseRail in ``recommendation`` mode as a drop-in for ListSkillTool.
No LLM call is made; relevance is computed from the pre-built scoring matrix
(``scoring_matrix_*.json`` files in the oracle dir) plus a lightweight pure-Python
query-term overlap against the matrix example prompts. The implementation has no
scikit-learn / pandas dependency.

The ``oracle_dir`` must be provided explicitly. When running inside JiuwenSwarm it
is resolved by ``_oracle_dir()`` in config_specs.py, which reads
``react.oracle_dir`` from config.yaml or the ``JIUWENSWARM_ORACLE_DIR`` env var.
There is no default path — if both are absent oracle_dir is None.
Passing ``oracle_dir=None`` means "not configured".

Graceful fallbacks (all return the full skill list with a ``mode`` marker):
  - oracle_dir is None (not configured)        -> fallback_no_oracle_dir
  - oracle_dir missing or has no matrix files  -> fallback_no_matrix
  - query is empty                             -> fallback_no_query
  - query produces no matches above threshold  -> fallback_no_match
  - any other runtime exception                -> fallback_error
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from openjiuwen.core.foundation.tool.base import Tool
from openjiuwen.core.single_agent.skills.skill_manager import Skill
from openjiuwen.harness.prompts.tools import build_tool_card
from openjiuwen.harness.tools.base_tool import ToolOutput

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens from *text* (pure Python)."""
    return set(_WORD_RE.findall(text.lower()))


def _load_matrix_rows(oracle_dir: Path) -> list[dict[str, Any]]:
    """Load per-(skill, example) score rows from scoring_matrix_*.json files.

    Supports the GEPA output format (``cross_eval`` rows under ``matrix`` /
    ``cross_eval`` keys) and a simple ``{skill: score}`` / ``[{name, score}]``
    mapping. Rows are flattened to ``{"skill", "example_input", "score"}``.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(oracle_dir.glob("scoring_matrix_*.json")):
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning("[RecommendSkillTool] failed to read %s: %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue

        skill_name = data.get("skill_name") or path.stem
        metrics = data.get("fitness_metrics") or []
        cross_eval = data.get("cross_eval")
        if isinstance(cross_eval, list):
            for row in cross_eval:
                if not isinstance(row, dict):
                    continue
                scores = row.get("scores") or {}
                metric_scores = [scores.get(m, 0.0) or 0.0 for m in metrics]
                mean = sum(metric_scores) / max(len(metric_scores), 1) if metric_scores else 0.0
                rows.append(
                    {
                        "skill": skill_name,
                        "example_input": row.get("example_input", ""),
                        "score": float(mean),
                    }
                )
            continue

        skills = data.get("skills")
        if isinstance(skills, dict):
            for name, score in skills.items():
                rows.append({"skill": name, "example_input": "", "score": float(score or 0.0)})
            continue
        if isinstance(skills, list):
            for item in skills:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("skill")
                if name:
                    rows.append(
                        {
                            "skill": name,
                            "example_input": item.get("example_input", ""),
                            "score": float(item.get("score", 0.0) or 0.0),
                        }
                    )
            continue

        # Plain {skill: score} mapping.
        try:
            for name, score in data.items():
                if name in ("run_id", "matrix", "cross_eval", "fitness_metrics", "skills"):
                    continue
                rows.append({"skill": name, "example_input": "", "score": float(score or 0.0)})
        except (TypeError, ValueError):
            continue
    return rows


def _rank_matrix_skills(rows: list[dict[str, Any]], query_terms: set[str]) -> list[dict[str, Any]]:
    """Aggregate matrix rows into per-skill scores ranked by relevance.

    A skill's score is its mean matrix score plus a pure-Python query-term
    overlap bonus against the matrix example prompts.
    """
    by_skill: dict[str, Dict[str, Any]] = {}
    for row in rows:
        entry = by_skill.setdefault(
            row["skill"], {"score_sum": 0.0, "count": 0, "examples": []}
        )
        entry["score_sum"] += float(row["score"])
        entry["count"] += 1
        if row["example_input"]:
            entry["examples"].append(str(row["example_input"]))

    ranked: list[dict[str, Any]] = []
    for name, entry in by_skill.items():
        matrix_score = entry["score_sum"] / max(entry["count"], 1)
        overlap = 0.0
        if query_terms and entry["examples"]:
            example_terms: set[str] = set()
            for example in entry["examples"]:
                example_terms |= _tokenize(example)
            if example_terms:
                overlap = len(query_terms & example_terms) / len(example_terms)
        ranked.append({"skill": name, "score": matrix_score + overlap})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


class RecommendSkillTool(Tool):
    """Recommend skills for a query using a pre-computed scoring matrix.

    The matrix is loaded lazily on first call and cached until the set of
    active skill names changes.
    """

    def __init__(
        self,
        get_skills: Callable[[], List[Skill]],
        oracle_dir: Optional[Path | str] = None,
        top_k: int = 10,
        score_threshold: float = 0.0,
        language: str = "cn",
        agent_id: Optional[str] = None,
    ) -> None:
        """Initialize the recommend_skill tool.

        Args:
            get_skills: Callable returning the current list of active Skill objects.
            oracle_dir: Directory containing ``scoring_matrix_*.json`` files.
                When None the tool falls back to returning all skills.
            top_k: Maximum number of recommendations to return.
            score_threshold: Minimum combined score for a recommendation to be kept.
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
        self._score_threshold = score_threshold
        self.language = language

        self._matrix: Optional[list[dict[str, Any]]] = None
        self._matrix_skill_key: Optional[frozenset[str]] = None

    async def invoke(self, inputs: Dict[str, Any], **kwargs) -> ToolOutput:
        """Invoke the recommend_skill tool."""
        oracle_dir_str = str(self._oracle_dir) if self._oracle_dir else None
        query = str(inputs.get("query", "") or "").strip()
        query_terms = _tokenize(query)

        if not query:
            return ToolOutput(
                success=True,
                data={"skills": self._dump_all_skills(), "mode": "fallback_no_query", "oracle_dir": oracle_dir_str},
            )
        if self._oracle_dir is None or not self._matrix_exists():
            logger.info(
                "[RecommendSkillTool] oracle_dir not configured or no matrix; falling back to all skills."
            )
            return ToolOutput(
                success=True,
                data={"skills": self._dump_all_skills(), "mode": "fallback_no_oracle_dir", "oracle_dir": oracle_dir_str},
            )

        try:
            rows = self._get_matrix_rows()
        except Exception as exc:  # noqa: BLE001 - graceful fallback contract
            logger.error("[RecommendSkillTool] failed to load matrix: %s", exc, exc_info=True)
            return ToolOutput(
                success=True,
                data={"skills": self._dump_all_skills(), "mode": "fallback_error", "oracle_dir": oracle_dir_str},
            )

        current_by_name: Dict[str, Skill] = {s.name: s for s in (self.get_skills() or [])}
        ranked = _rank_matrix_skills(rows, query_terms)
        results: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for rec in ranked:
            name = rec["skill"]
            if name not in current_by_name or name in seen:
                continue
            if rec["score"] < self._score_threshold:
                continue
            seen.add(name)
            skill = current_by_name[name]
            item = skill.asdict(include_directory=True)
            item["skill_md_path"] = str(Path(skill.directory) / "SKILL.md")
            item["score"] = round(rec["score"], 4)
            results.append(item)
            if len(results) >= self._top_k:
                break

        if not results:
            return ToolOutput(
                success=True,
                data={"skills": self._dump_all_skills(), "mode": "fallback_no_match", "oracle_dir": oracle_dir_str},
            )
        return ToolOutput(
            success=True,
            data={"skills": results, "mode": "recommendation", "oracle_dir": oracle_dir_str},
        )

    async def stream(self, inputs: Dict[str, Any], **kwargs) -> AsyncIterator[Any]:
        """Stream is not supported for this tool."""
        if False:  # pragma: no cover - placeholder for the required async generator
            yield None

    # -- internal helpers -----------------------------------------------------

    def _matrix_exists(self) -> bool:
        """Return True if the oracle dir is configured, exists, and has matrix files."""
        if self._oracle_dir is None or not self._oracle_dir.exists():
            return False
        return any(self._oracle_dir.glob("scoring_matrix_*.json"))

    def _get_matrix_rows(self) -> list[dict[str, Any]]:
        """Return the loaded matrix rows, cached by the active skill set."""
        current_key = frozenset(s.name for s in (self.get_skills() or []))
        if self._matrix is not None and self._matrix_skill_key == current_key:
            return self._matrix
        self._matrix = _load_matrix_rows(self._oracle_dir or Path())
        self._matrix_skill_key = current_key
        return self._matrix

    def _dump_all_skills(self) -> List[Dict[str, Any]]:
        """Dump all current enabled skills as serializable dicts."""
        results: List[Dict[str, Any]] = []
        for skill in self.get_skills() or []:
            item = skill.asdict(include_directory=True)
            item["skill_md_path"] = str(Path(skill.directory) / "SKILL.md")
            results.append(item)
        return results


__all__ = ["RecommendSkillTool"]
