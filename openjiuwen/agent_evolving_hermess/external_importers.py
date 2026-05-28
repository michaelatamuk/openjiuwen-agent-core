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
    from openjiuwen.agent_evolving_hermess.dataset_builder import EvalDataset, EvalExample

# ── Secret detection (same patterns as Hermess) ───────────────────────────────

SECRET_PATTERNS = re.compile(
    r"("
    r"sk-ant-api\S+"
    r"|sk-or-v1-\S+"
    r"|sk-\S{20,}"
    r"|ghp_\S+"
    r"|ghu_\S+"
    r"|xoxb-\S+"
    r"|xapp-\S+"
    r"|ntn_\S+"
    r"|AKIA[0-9A-Z]{16}"
    r"|Bearer\s+\S{20,}"
    r"|-----BEGIN\s+(RSA\s+)?PRIVATE\sKEY-----"
    r"|ANTHROPIC_API_KEY"
    r"|OPENAI_API_KEY"
    r"|OPENROUTER_API_KEY"
    r"|JIUWEN_API_KEY"
    r"|SLACK_BOT_TOKEN"
    r"|GITHUB_TOKEN"
    r"|AWS_SECRET_ACCESS_KEY"
    r"|DATABASE_URL"
    r"|\bpassword\s*[=:]\s*\S+"
    r"|\bsecret\s*[=:]\s*\S+"
    r"|\btoken\s*[=:]\s*\S{10,}"
    r")",
    re.IGNORECASE,
)


def _contains_secret(text: str) -> bool:
    return bool(SECRET_PATTERNS.search(text))


# ── Jiuwen session importer ───────────────────────────────────────────────────


class JiuwenSessionImporter:
    """Import user/assistant pairs from ~/.jiuwen/sessions/*.json"""

    SESSION_DIR = Path.home() / ".jiuwen" / "sessions"

    @staticmethod
    def extract_messages(limit: int = 0) -> List[Dict]:
        if not JiuwenSessionImporter.SESSION_DIR.exists():
            return []
        messages = []
        session_files = sorted(
            JiuwenSessionImporter.SESSION_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for sf in session_files:
            try:
                data = json.loads(sf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            msg_list = data.get("messages", [])
            session_id = data.get("session_id", sf.stem)
            for i, msg in enumerate(msg_list):
                if msg.get("role") != "user":
                    continue
                user_text = msg.get("content", "")
                if not user_text or len(user_text) < 10:
                    continue
                if _contains_secret(user_text):
                    continue
                assistant_text = ""
                for j in range(i + 1, len(msg_list)):
                    if msg_list[j].get("role") == "assistant":
                        c = msg_list[j].get("content", "")
                        if c:
                            assistant_text = c
                            break
                    elif msg_list[j].get("role") == "user":
                        break
                if assistant_text and _contains_secret(assistant_text):
                    continue
                messages.append(
                    {
                        "source": "jiuwen",
                        "task_input": user_text,
                        "assistant_response": assistant_text,
                        "session_id": session_id,
                    }
                )
                if limit and len(messages) >= limit:
                    return messages
        return messages


# ── Claude Code importer (same as Hermess) ────────────────────────────────────


class ClaudeCodeImporter:
    HISTORY_PATH = Path.home() / ".claude" / "history.jsonl"

    @staticmethod
    def extract_messages(limit: int = 0) -> List[Dict]:
        if not ClaudeCodeImporter.HISTORY_PATH.exists():
            return []
        messages = []
        with open(ClaudeCodeImporter.HISTORY_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = entry.get("display", "")
                if not text or len(text) < 10 or _contains_secret(text):
                    continue
                messages.append(
                    {
                        "source": "claude-code",
                        "task_input": text,
                        "session_id": entry.get("sessionId", ""),
                    }
                )
                if limit and len(messages) >= limit:
                    break
        return messages


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
        from openjiuwen.agent_evolving_hermess.dataset_builder import EvalExample

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


def build_dataset_from_external(
    skill_name: str,
    skill_text: str,
    sources: List[str],
    output_path: Path,
    model: str,
    max_examples: int = 50,
) -> "EvalDataset":
    from openjiuwen.agent_evolving_hermess.dataset_builder import EvalDataset

    all_messages: List[Dict] = []
    importers = {
        "jiuwen": ("Jiuwen sessions", JiuwenSessionImporter),
        "claude-code": ("Claude Code history", ClaudeCodeImporter),
    }
    for source in sources:
        if source in importers:
            _label, cls = importers[source]
            msgs = cls.extract_messages()
            all_messages.extend(msgs)
    if not all_messages:
        return EvalDataset()
    rf = RelevanceFilter(model=model)
    examples = rf.filter_and_score(all_messages, skill_name, skill_text, max_examples)
    if not examples:
        return EvalDataset()
    random.shuffle(examples)
    n = len(examples)
    n_train = max(1, int(n * 0.5))
    n_val = max(1, int(n * 0.25))
    ds = EvalDataset(
        train=examples[:n_train],
        val=examples[n_train: n_train + n_val],
        holdout=examples[n_train + n_val:],
    )
    ds.save(output_path)
    return ds
