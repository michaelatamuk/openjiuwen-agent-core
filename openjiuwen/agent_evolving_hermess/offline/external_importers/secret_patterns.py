# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Session log mining for eval dataset construction.

Mirrors hermes-agent-self-evolution evolution/core/external_importers.py.
Key Jiuwen adaptation: JiuwenSessionImporter reads ~/.jiuwen/sessions/*.json
(same format as HermesSessionImporter but different path).
"""
import re

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
