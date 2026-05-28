# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Write provenance / audit metadata for skill and memory writes.

Mirrors Hermess build_memory_write_metadata().
"""
from __future__ import annotations

import os
from typing import Any, Dict


def make_write_metadata(
    *,
    write_origin: str = "background_review",
    execution_context: str = "background_review",
    session_id: str = "",
    parent_session_id: str = "",
    platform: str = "",
) -> Dict[str, Any]:
    """Build provenance metadata dict for skill/memory writes."""
    platform = platform or os.environ.get("JIUWEN_SESSION_SOURCE", "sdk")
    result: Dict[str, Any] = {
        "write_origin": write_origin,
        "execution_context": execution_context,
        "tool_name": "background_review",
    }
    if session_id:
        result["session_id"] = session_id
    if parent_session_id:
        result["parent_session_id"] = parent_session_id
    if platform:
        result["platform"] = platform
    return result
