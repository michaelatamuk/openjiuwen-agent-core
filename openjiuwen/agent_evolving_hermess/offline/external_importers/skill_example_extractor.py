# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""LLM-based extractor that converts raw session messages into EvalExamples.

For each candidate message the extractor calls an LLM to:
  1. Decide whether the message is relevant to the target skill.
  2. Generate a proper expected_behavior rubric (not a copy of the actual
     response — a description of what ideal behaviour looks like for that
     task and skill).

Mirrors hermes-agent-self-evolution evolution/core/external_importers.py.
"""
from __future__ import annotations

import json
import random
import re
from typing import TYPE_CHECKING, Dict, List, Optional

import dspy

if TYPE_CHECKING:
    from openjiuwen.agent_evolving_hermess.offline.dataset_builder import EvalExample


# ── Keyword pre-filter (cheap, runs before any LLM call) ─────────────────────


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


def _parse_example_json(text: str) -> Optional[Dict]:
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


# ── Main extractor ────────────────────────────────────────────────────────────


class SkillExampleExtractor:
    """Convert raw session messages into EvalExamples with LLM-generated rubrics.

    For each candidate message:
      - Checks relevance to the skill (keyword pre-filter, then LLM).
      - Calls GenerateExpectedBehavior to produce a proper expected_behavior
        rubric informed by the actual assistant response but not identical to it.

    Usage::

        extractor = SkillExampleExtractor(model="openai/gpt-4o")
        examples = extractor.extract_examples(messages, skill_name, skill_text)
    """

    class GenerateExpectedBehavior(dspy.Signature):
        """Decide if a conversation turn is relevant to a skill and, if so,
        generate an expected_behavior rubric describing what ideal behaviour
        looks like for that task.

        The assistant_response is provided as context only — do NOT copy it
        verbatim.  Write a rubric that describes quality criteria.

        Return JSON: {relevant: bool, expected_behavior: str, difficulty: str, category: str}
        """

        skill_name: str = dspy.InputField(desc="Name of the skill")
        skill_description: str = dspy.InputField(desc="First 800 chars of skill file")
        user_message: str = dspy.InputField(desc="The user's message")
        assistant_response: str = dspy.InputField(
            desc="The assistant's actual response (context only — do not copy)"
        )
        example_json: str = dspy.OutputField(
            desc="JSON: {relevant, expected_behavior, difficulty, category}"
        )

    def __init__(self, model: str):
        self.generator = dspy.ChainOfThought(self.GenerateExpectedBehavior)
        self.model = model

    def extract_examples(
        self,
        messages: List[Dict],
        skill_name: str,
        skill_text: str,
        max_examples: int = 50,
    ) -> "List[EvalExample]":
        """Extract EvalExamples from raw session messages.

        Args:
            messages:     List of dicts with keys: task_input, assistant_response, source.
            skill_name:   Name of the skill being evolved.
            skill_text:   Full text of the skill file (used for keyword matching).
            max_examples: Maximum number of EvalExamples to return.

        Returns:
            List of EvalExamples with LLM-generated expected_behavior rubrics.
        """
        from openjiuwen.agent_evolving_hermess.offline.dataset_builder import EvalExample

        skill_desc = skill_text[:800]
        messages = [m for m in messages if m.get("task_input") and m.get("source")]

        # Keyword pre-filter — cheap, runs before any LLM call
        candidates = [
            m for m in messages
            if _is_relevant_to_skill(m["task_input"], skill_name, skill_text)
        ]
        # Top up with remaining messages if not enough keyword hits
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
                    result = self.generator(
                        skill_name=skill_name,
                        skill_description=skill_desc,
                        user_message=msg["task_input"][:1000],
                        assistant_response=msg.get("assistant_response", "")[:1000],
                    )
                parsed = _parse_example_json(result.example_json)
                if parsed and parsed.get("relevant"):
                    examples.append(
                        EvalExample(
                            task_input=msg["task_input"][:2000],
                            expected_behavior=parsed.get("expected_behavior", ""),
                            difficulty=parsed.get("difficulty", "medium"),
                            category=parsed.get("category", "general"),
                            source=msg["source"],
                        )
                    )
            except Exception:
                pass
            if len(examples) >= max_examples:
                break
        return examples
