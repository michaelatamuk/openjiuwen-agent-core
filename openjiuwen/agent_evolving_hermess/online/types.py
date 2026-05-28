# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared data types for agent_evolving_hermess.

Contains enums and dataclasses used by both the online
(BackgroundReviewRail) and offline (GEPA skill evolver) tracks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ReviewMode(str, Enum):
    """Which stores the background review pass should update."""
    MEMORY_ONLY = "memory_only"
    SKILLS_ONLY = "skills_only"
    COMBINED = "combined"


@dataclass
class ReviewTrigger:
    """Snapshot of what triggered a background review pass."""
    mode: ReviewMode
    user_turn_count: int
    tool_iter_count: int
    session_id: str


@dataclass
class ReviewAction:
    """One write action successfully emitted by background review."""
    action_type: str
    """One of: skill_create, skill_edit, skill_patch, skill_delete,
    memory_add, memory_replace, memory_remove."""

    target_name: str
    """Skill name, or 'memory' / 'user' for memory actions."""

    summary: str
    """Human-readable one-line description, e.g. 'Updated git-review'."""

    write_origin: str = "background_review"
    execution_context: str = "background_review"
    session_id: str = ""
    parent_session_id: str = ""


@dataclass
class ReviewResult:
    """Result returned by a completed background review pass."""
    trigger: ReviewTrigger
    actions: List[ReviewAction] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    summary_line: str = ""
    """Human-readable summary, e.g. 'Updated git-review · Added memory entry'."""
