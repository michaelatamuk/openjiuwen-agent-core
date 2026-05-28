# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""BackgroundReviewRail — Jiuwen implementation of Hermess background review.

Priority: 70
  (below existing SkillEvolutionRail at 80 and HealingRail at 90;
   above EvolutionRail base at 60; runs AFTER primary rails have finished)

Trigger logic (mirrors Hermess exactly):
  _user_turn_count   incremented in after_model_call when a non-tool
                     assistant message is observed.
  _tool_iter_count   incremented in after_tool_call on every tool completion.

Background task (mirrors Hermess _spawn_background_review):
  asyncio.create_task(run_background_review(...)) spawned in after_invoke.
  Task is awaited in the NEXT after_invoke if still running (serialisation).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from openjiuwen.core.single_agent.rail.base import AgentCallbackContext
from openjiuwen.harness.rails.base import DeepAgentRail

from openjiuwen.agent_evolving_hermess.config import BackgroundReviewConfig
from openjiuwen.agent_evolving_hermess.review_executor import run_background_review
from openjiuwen.agent_evolving_hermess.types import ReviewMode, ReviewResult, ReviewTrigger

logger = logging.getLogger(__name__)


class BackgroundReviewRail(DeepAgentRail):
    """Hermess-style background review rail for Jiuwen.

    After every N tool iterations or M user turns, spawns an async background
    task that reads the full conversation and uses an LLM to write targeted
    updates to SKILL.md files and memory stores.

    Does NOT touch existing agent_evolving or agent_healing.
    """

    priority: int = 70

    def __init__(self, config: Optional[BackgroundReviewConfig] = None):
        super().__init__()
        self._config: BackgroundReviewConfig = config or BackgroundReviewConfig()
        self._user_turn_count: int = 0
        self._tool_iter_count: int = 0
        self._review_task: Optional[asyncio.Task] = None
        self._last_result: Optional[ReviewResult] = None
        # Snapshot of messages captured at end of invoke (before context cleared)
        self._messages_snapshot: List[Dict[str, Any]] = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_model(self, ctx: AgentCallbackContext) -> str:
        """Resolve LLM model for review. Falls back to agent model."""
        if self._config.review_model:
            return self._config.review_model
        # Try to read model name from agent
        agent = ctx.agent
        for attr in ("model", "_model", "model_name"):
            val = getattr(agent, attr, None)
            if isinstance(val, str) and val:
                return val
        return "gpt-4o-mini"  # safe fallback

    def _get_session_id(self, ctx: AgentCallbackContext) -> str:
        if ctx.session:
            return ctx.session.get_session_id() or ""
        return ""

    def _capture_messages(self, ctx: AgentCallbackContext) -> List[Dict[str, Any]]:
        """Snapshot the current conversation messages as plain dicts."""
        if ctx.context is None:
            return []
        msgs = ctx.context.get_messages()
        result = []
        for m in msgs:
            role = getattr(m, "role", "unknown")
            content = getattr(m, "content", "")
            result.append({"role": role, "content": content})
        return result

    # ── Hooks ─────────────────────────────────────────────────────────────────

    async def after_model_call(self, ctx: AgentCallbackContext) -> None:
        """Increment user-turn counter when assistant produces a text response."""
        if not self._config.enabled:
            return
        # Detect a non-tool assistant message (mirrors Hermess _turns_since_memory logic)
        msg = getattr(ctx.inputs, "response_message", None) or getattr(
            ctx.inputs, "message", None
        )
        if msg:
            role = getattr(msg, "role", "")
            tool_calls = getattr(msg, "tool_calls", None)
            if role == "assistant" and not tool_calls:
                self._user_turn_count += 1

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        """Increment tool-iteration counter on every tool completion."""
        if not self._config.enabled:
            return
        if self._config.skill_nudge_interval > 0:
            self._tool_iter_count += 1

    async def after_invoke(self, ctx: AgentCallbackContext) -> None:
        """At the end of each invoke:

        1. Wait for any in-flight review task (serialisation).
        2. Check trigger conditions.
        3. Snapshot messages and spawn background review task.
        """
        if not self._config.enabled:
            return

        # Serialise: wait for previous review to finish before starting a new one
        if self._review_task and not self._review_task.done():
            try:
                await asyncio.wait_for(self._review_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Best-effort; don't block current invoke

        # Determine what to review
        should_review_memory = (
            self._config.memory_nudge_interval > 0
            and self._user_turn_count >= self._config.memory_nudge_interval
        )
        should_review_skills = (
            self._config.skill_nudge_interval > 0
            and self._tool_iter_count >= self._config.skill_nudge_interval
        )

        if not (should_review_memory or should_review_skills):
            return

        # Determine mode
        if should_review_memory and should_review_skills:
            mode = ReviewMode.COMBINED
        elif should_review_memory:
            mode = ReviewMode.MEMORY_ONLY
        else:
            mode = ReviewMode.SKILLS_ONLY

        # Reset counters (mirrors Hermess reset-on-fire)
        if should_review_memory:
            self._user_turn_count = 0
        if should_review_skills:
            self._tool_iter_count = 0

        # Snapshot messages before they may be cleared
        messages_snapshot = self._capture_messages(ctx)
        session_id = self._get_session_id(ctx)
        model = self._get_model(ctx)

        trigger = ReviewTrigger(
            mode=mode,
            user_turn_count=self._user_turn_count,
            tool_iter_count=self._tool_iter_count,
            session_id=session_id,
        )

        # Spawn background task — does NOT block the agent
        self._review_task = asyncio.create_task(
            self._run_review(messages_snapshot, trigger, model, session_id)
        )
        logger.debug(
            "BackgroundReviewRail: spawned review task [mode=%s session=%s]",
            mode.value,
            session_id,
        )

    # ── Background task ───────────────────────────────────────────────────────

    async def _run_review(
        self,
        messages_snapshot: List[Dict[str, Any]],
        trigger: ReviewTrigger,
        model: str,
        session_id: str,
    ) -> None:
        """Background coroutine. Mirrors Hermess _run_review_in_thread()."""
        try:
            result = await run_background_review(
                messages_snapshot=messages_snapshot,
                trigger=trigger,
                config=self._config,
                model=model,
                session_id=session_id,
            )
            self._last_result = result
            if result.actions:
                logger.info(
                    "BackgroundReview completed: %s [%.1fs]",
                    result.summary_line,
                    result.duration_seconds,
                )
            if result.error:
                logger.warning("BackgroundReview error: %s", result.error)
        except Exception as e:
            logger.error("BackgroundReviewRail background task crashed: %s", e)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def uninit(self, agent) -> None:
        """Cancel any in-flight review task on rail teardown."""
        if self._review_task and not self._review_task.done():
            self._review_task.cancel()

    # ── Public accessors (for tests and observability) ────────────────────────

    def last_review_result(self) -> Optional[ReviewResult]:
        """Return the result of the most recently completed review, or None."""
        return self._last_result

    def pending_counts(self) -> Dict[str, int]:
        """Return current trigger counters (useful for debugging)."""
        return {
            "user_turns_since_review": self._user_turn_count,
            "tool_iters_since_review": self._tool_iter_count,
        }
