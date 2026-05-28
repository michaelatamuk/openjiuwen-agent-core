# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Import real conversation trajectories as eval dataset examples.

Supports two sources:
  1. A folder of saved trajectory JSON files
     (each file exported from agent_evolving.agent_rl trajectory store).
  2. A list of in-memory Trajectory objects
     (any object with a ``to_messages()`` method, e.g. from InMemoryTrajectoryRegistry).

In both cases the importer produces {task_input, assistant_response, source} dicts
in the same format as JiuwenSessionImporter, so they feed directly into
SkillExampleExtractor which generates proper expected_behavior rubrics.

Only steps with reward >= min_reward are used.  When no rewards are recorded
(reward is None on all steps) all steps are used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _extract_messages_from_steps(steps: List[Dict[str, Any]], min_reward: float) -> List[Dict[str, Any]]:
    """Extract user/assistant pairs from raw step dicts (deserialized JSON).

    A step is used when:
      - kind == "llm"
      - reward is None (no reward signal) OR reward >= min_reward
    """
    pairs: List[Dict[str, Any]] = []
    for step in steps:
        if step.get("kind") != "llm":
            continue
        reward = step.get("reward")
        if reward is not None and reward < min_reward:
            continue
        detail = step.get("detail") or {}
        messages: List[Dict[str, Any]] = detail.get("messages") or []
        response: Optional[Dict[str, Any]] = detail.get("response")

        # Walk messages to find user → assistant pairs
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            user_text = msg.get("content", "")
            if not user_text or not isinstance(user_text, str) or len(user_text) < 10:
                continue
            # Look for the next assistant message
            assistant_text = ""
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") == "assistant":
                    c = messages[j].get("content", "")
                    if c and isinstance(c, str):
                        assistant_text = c
                    break
                if messages[j].get("role") == "user":
                    break
            # Fall back to the response object from the step
            if not assistant_text and response:
                c = response.get("content", "")
                if c and isinstance(c, str):
                    assistant_text = c
            if user_text:
                pairs.append({
                    "task_input": user_text,
                    "assistant_response": assistant_text,
                    "source": "trajectory",
                })
    return pairs


class TrajectoryImporter:
    """Import real conversation trajectories as session-message dicts.

    Usage (from saved JSON files)::

        importer = TrajectoryImporter(
            trajectory_dir=Path("/path/to/trajectories"),
            min_reward=0.5,
        )
        messages = importer.extract_messages()

    Usage (from in-memory Trajectory objects)::

        importer = TrajectoryImporter(min_reward=0.0)
        messages = importer.extract_messages_from_objects(trajectory_list)
    """

    def __init__(
        self,
        trajectory_dir: Optional[Path] = None,
        min_reward: float = 0.0,
    ) -> None:
        self.trajectory_dir = trajectory_dir
        self.min_reward = min_reward

    def extract_messages(self, limit: int = 0) -> List[Dict[str, Any]]:
        """Load trajectory JSON files from trajectory_dir and extract message pairs.

        Args:
            limit: Maximum number of message pairs to return (0 = unlimited).

        Returns:
            List of {task_input, assistant_response, source} dicts.
        """
        if not self.trajectory_dir or not Path(self.trajectory_dir).exists():
            return []

        messages: List[Dict[str, Any]] = []
        traj_files = sorted(
            Path(self.trajectory_dir).glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for tf in traj_files:
            try:
                data = json.loads(tf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            # Support both a single trajectory dict and a list of them
            if isinstance(data, list):
                trajectories = data
            elif isinstance(data, dict):
                trajectories = [data]
            else:
                continue
            for traj in trajectories:
                steps = traj.get("steps") or []
                pairs = _extract_messages_from_steps(steps, self.min_reward)
                messages.extend(pairs)
                if limit and len(messages) >= limit:
                    return messages[:limit]

        return messages[:limit] if limit else messages

    def extract_messages_from_objects(
        self,
        trajectories: List[Any],
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """Extract message pairs from in-memory Trajectory objects.

        Accepts any object with a ``to_messages()`` method (e.g.
        openjiuwen.agent_evolving.trajectory.types.Trajectory).

        Args:
            trajectories: List of Trajectory-like objects.
            limit:        Maximum number of message pairs to return (0 = unlimited).

        Returns:
            List of {task_input, assistant_response, source} dicts.
        """
        messages: List[Dict[str, Any]] = []
        for traj in trajectories:
            raw_messages: List[Dict[str, Any]] = []
            if hasattr(traj, "to_messages") and callable(traj.to_messages):
                raw_messages = traj.to_messages()
            else:
                continue  # not a recognised Trajectory object

            for i, msg in enumerate(raw_messages):
                if msg.get("role") != "user":
                    continue
                user_text = msg.get("content", "")
                if not user_text or not isinstance(user_text, str) or len(user_text) < 10:
                    continue
                assistant_text = ""
                for j in range(i + 1, len(raw_messages)):
                    if raw_messages[j].get("role") == "assistant":
                        c = raw_messages[j].get("content", "")
                        if c and isinstance(c, str):
                            assistant_text = c
                        break
                    if raw_messages[j].get("role") == "user":
                        break
                messages.append({
                    "task_input": user_text,
                    "assistant_response": assistant_text,
                    "source": "trajectory",
                })
                if limit and len(messages) >= limit:
                    return messages

        return messages
