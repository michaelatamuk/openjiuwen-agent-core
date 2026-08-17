# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for RecommendSkillTool (recommendation skill mode)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openjiuwen.core.single_agent.skills.skill_manager import Skill
from openjiuwen.harness.tools.skills.recommend_skill import RecommendSkillTool


def _skill(name: str) -> Skill:
    return Skill(name=name, description=f"desc for {name}", directory=Path(f"/skills/{name}"))


def _write_matrix(oracle_dir: Path) -> None:
    """Write a GEPA-style scoring matrix with two skills."""
    data = {
        "run_id": "run-1",
        "skill_name": "alpha",
        "fitness_metrics": ["bag_of_words", "f1"],
        "cross_eval": [
            {"example_id": "e1", "example_input": "search web results and summarize them", "scores": {"bag_of_words": 0.9, "f1": 0.8}},
            {"example_id": "e2", "example_input": "fetch a webpage and extract main text", "scores": {"bag_of_words": 0.7, "f1": 0.6}},
        ],
    }
    (oracle_dir / "scoring_matrix_alpha.json").write_text(json.dumps(data), encoding="utf-8")


def _make_tool(oracle_dir: Path | None, skills: list[Skill]) -> RecommendSkillTool:
    return RecommendSkillTool(
        get_skills=lambda: skills,
        oracle_dir=str(oracle_dir) if oracle_dir else None,
    )


@pytest.mark.asyncio
async def test_recommend_skill_returns_ranked_matrix_skills(tmp_path: Path) -> None:
    """A query matching the matrix examples returns ranked skills."""
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    _write_matrix(oracle)
    skills = [_skill("alpha"), _skill("beta")]
    tool = _make_tool(oracle, skills)

    result = await tool.invoke({"query": "search the web and summarize"})

    assert result.success is True
    data = result.data
    assert data["mode"] == "recommendation"
    names = [item["name"] for item in data["skills"]]
    assert "alpha" in names
    # beta is not in the matrix and must not leak into the recommendations.
    assert "beta" not in names


@pytest.mark.asyncio
async def test_recommend_skill_falls_back_without_oracle_dir(tmp_path: Path) -> None:
    """Without an oracle dir the tool returns all skills (fallback mode)."""
    skills = [_skill("alpha"), _skill("beta")]
    tool = _make_tool(None, skills)

    result = await tool.invoke({"query": "anything"})

    assert result.success is True
    assert result.data["mode"] == "fallback_no_oracle_dir"
    assert {item["name"] for item in result.data["skills"]} == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_recommend_skill_falls_back_on_empty_query(tmp_path: Path) -> None:
    """An empty query returns all skills with the no-query marker."""
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    _write_matrix(oracle)
    skills = [_skill("alpha")]
    tool = _make_tool(oracle, skills)

    result = await tool.invoke({"query": ""})

    assert result.success is True
    assert result.data["mode"] == "fallback_no_query"


@pytest.mark.asyncio
async def test_recommend_skill_handles_simple_skill_map(tmp_path: Path) -> None:
    """A simple {skill: score} mapping is also supported."""
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    (oracle / "scoring_matrix_simple.json").write_text(
        json.dumps({"alpha": 0.9, "gamma": 0.5}), encoding="utf-8"
    )
    skills = [_skill("alpha"), _skill("gamma"), _skill("beta")]
    tool = _make_tool(oracle, skills)

    result = await tool.invoke({"query": "anything"})

    assert result.success is True
    assert result.data["mode"] == "recommendation"
    assert {item["name"] for item in result.data["skills"]} == {"alpha", "gamma"}
