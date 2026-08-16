# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Phase-0 skill recommender (collaborative, offline, TF-IDF).

Vendored from agent_evolving/examples/offline/sage/skill_recommender (Phase 0 only).
Do NOT add Phase 1/2/3 code here — those phases are not used by JiuwenSwarm.

Runtime dependencies (optional — graceful fallback when absent):
    pip install 'openjiuwen[oracle]'   # installs scikit-learn + pandas

Imports are intentionally NOT performed here at package level to avoid
ImportError at startup when scikit-learn / pandas are not installed.
RecommendSkillTool catches ImportError and falls back to listing all skills.
"""
