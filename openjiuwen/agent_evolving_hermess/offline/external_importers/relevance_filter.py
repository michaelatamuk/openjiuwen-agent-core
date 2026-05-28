# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Session log mining for eval dataset construction.

Mirrors hermes-agent-self-evolution evolution/core/external_importers.py.
Key Jiuwen adaptation: JiuwenSessionImporter reads ~/.jiuwen/sessions/*.json
(same format as HermesSessionImporter but different path).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import dspy

if TYPE_CHECKING:
    from openjiuwen.agent_evolving_hermess.offline.dataset_builder import EvalDataset, EvalExample


# ── Relevance filter (mirrors Hermess exactly) ────────────────────────────────


def _is_relevant_to_skill(text: str, skill_name: str, skill_text: str) -> bool:
    text_lower = text.lower()
    skill_lower = skill_name.lower().replace("-", " ").replace("_", " ")
    if skill_lower in text_lower:
        return True
    for word in skill_lower.split():
        if len(word) > 3 and word in text_lower:
            return True
    skill_keywords = set()
    for word in skill_text[:500].lower().split():
        word = re.sub(r"[^a-z]", "", word)
        if len(word) > 4:
            skill_keywords.add(word)
    message_words = set(re.sub(r"[^a-z\s]", "", text_lower).split())
    return len(message_words & skill_keywords) >= 2


def _parse_scoring_json(text: str) -> Optional[Dict]:
    if not text:
        return None
    try:
        r = json.loads(text)
        if isinstance(r, dict):
            return r
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escape_next = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class RelevanceFilter:
    class ScoreRelevance(dspy.Signature):
        """Score whether a user message is relevant to a skill.

        Return JSON: {relevant: bool, expected_behavior: str, difficulty: str, category: str}
        """

        skill_name: str = dspy.InputField(desc="Name of the skill")
        skill_description: str = dspy.InputField(
            desc="First 800 chars of skill file"
        )
        user_message: str = dspy.InputField(desc="The user's message")
        assistant_response: str = dspy.InputField(
            desc="The assistant's response (may be empty)"
        )
        scoring: str = dspy.OutputField(
            desc="JSON: {relevant, expected_behavior, difficulty, category}"
        )

    def __init__(self, model: str):
        self.scorer = dspy.ChainOfThought(self.ScoreRelevance)
        self.model = model

    def filter_and_score(
        self,
        messages: List[Dict],
        skill_name: str,
        skill_text: str,
        max_examples: int = 50,
    ) -> "List[EvalExample]":
        from openjiuwen.agent_evolving_hermess.offline.dataset_builder import EvalExample

        skill_desc = skill_text[:800]
        messages = [m for m in messages if m.get("task_input") and m.get("source")]
        candidates = [
            m
            for m in messages
            if _is_relevant_to_skill(m["task_input"], skill_name, skill_text)
        ]
        if len(candidates) < max_examples:
            extra = [m for m in messages if m not in candidates]
            random.shuffle(extra)
            candidates.extend(extra[: max_examples * 2])
        candidates = candidates[: max_examples * 3]

        examples = []
        lm = dspy.LM(self.model)
        for msg in candidates:
            try:
                with dspy.context(lm=lm):
                    result = self.scorer(
                        skill_name=skill_name,
                        skill_description=skill_desc,
                        user_message=msg["task_input"][:1000],
                        assistant_response=msg.get("assistant_response", "")[:1000],
                    )
                scoring = _parse_scoring_json(result.scoring)
                if scoring and scoring.get("relevant"):
                    examples.append(
                        EvalExample(
                            task_input=msg["task_input"][:2000],
                            expected_behavior=scoring.get("expected_behavior", ""),
                            difficulty=scoring.get("difficulty", "medium"),
                            category=scoring.get("category", "general"),
                            source=msg["source"],
                        )
                    )
            except Exception:
                pass
            if len(examples) >= max_examples:
                break
        return examples
