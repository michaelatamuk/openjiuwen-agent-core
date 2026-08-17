# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bilingual description and input params for the RecommendSkill tool."""

from __future__ import annotations

from typing import Any, Dict

from openjiuwen.harness.prompts.tools.base import (
    ToolMetadataProvider,
)

DESCRIPTION: Dict[str, str] = {
    "cn": "根据当前任务查询相关技能，按相关性评分排序返回。使用预计算评分矩阵，无需额外 LLM 调用。",
    "en": "Query relevant skills for the current task, returned sorted by relevance score. "
          "Uses a pre-computed scoring matrix — no extra LLM call required.",
}

RECOMMEND_SKILL_PARAMS: Dict[str, Dict[str, str]] = {
    "query": {
        "cn": "当前用户任务或问题描述，用于检索最相关的技能。",
        "en": "Current user task or question description used to retrieve the most relevant skills.",
    },
}


def get_recommend_skill_input_params(language: str = "cn") -> Dict[str, Any]:
    """Return the full JSON Schema for recommend_skill tool input_params."""
    p = RECOMMEND_SKILL_PARAMS
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": p["query"].get(language, p["query"]["cn"]),
            },
        },
        "required": ["query"],
    }


class RecommendSkillMetadataProvider(ToolMetadataProvider):
    """RecommendSkillTool 工具的元数据 provider。"""

    def get_name(self) -> str:
        return "recommend_skill"

    def get_description(self, language: str = "cn") -> str:
        return DESCRIPTION.get(language, DESCRIPTION["cn"])

    def get_input_params(self, language: str = "cn") -> Dict[str, Any]:
        return get_recommend_skill_input_params(language)


__all__ = ["RecommendSkillMetadataProvider", "get_recommend_skill_input_params"]
