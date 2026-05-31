# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Session log mining for eval dataset construction.

Mirrors hermes-agent-self-evolution evolution/core/external_importers.py.
Key Jiuwen adaptation: JiuwenSessionImporter reads ~/.jiuwen/sessions/*.json
(same format as HermesSessionImporter but different path).
"""
import re

# ── Secret detection ──────────────────────────────────────────────────────────
# Expanded from hermes-agent-self-evolution evolution/core/external_importers.py
# to cover more provider token formats, private key types, database URIs,
# and common env-var names that indicate credential material.

SECRET_PATTERNS = re.compile(
    r"("
    # ── LLM provider tokens ───────────────────────────────────────────────────
    r"sk-ant-api\S+"                        # Anthropic
    r"|sk-or-v1-\S+"                        # OpenRouter
    r"|sk-\S{20,}"                          # OpenAI / generic sk- tokens
    r"|hf_[A-Za-z0-9]{30,}"               # Hugging Face
    # ── Source control ────────────────────────────────────────────────────────
    r"|ghp_\S+"                             # GitHub personal access token
    r"|ghu_\S+"                             # GitHub user token
    r"|ghs_\S+"                             # GitHub server token
    r"|ghr_\S+"                             # GitHub refresh token
    # ── Cloud / infra ─────────────────────────────────────────────────────────
    r"|AKIA[0-9A-Z]{16}"                    # AWS access key ID (permanent)
    r"|ASIA[0-9A-Z]{16}"                    # AWS access key ID (temporary STS)
    r"|AIza[0-9A-Za-z\-_]{35}"            # Google API key
    r"|hvs\.\S{20,}"                        # HashiCorp Vault token
    # ── Payment / messaging ───────────────────────────────────────────────────
    r"|sk_live_[0-9a-zA-Z]{24,}"           # Stripe secret key
    r"|rk_live_[0-9a-zA-Z]{24,}"           # Stripe restricted key
    r"|SG\.[A-Za-z0-9\-_]{22}\.[A-Za-z0-9\-_]{43}"  # SendGrid
    r"|AC[a-f0-9]{32}"                      # Twilio account SID
    # ── Chat / collaboration ──────────────────────────────────────────────────
    r"|xoxb-\S+"                            # Slack bot token
    r"|xapp-\S+"                            # Slack app token
    r"|xoxp-\S+"                            # Slack user token
    r"|xoxr-\S+"                            # Slack refresh token
    r"|ntn_\S+"                             # Notion token
    # ── Package registry auth ─────────────────────────────────────────────────
    r"|pypi-[A-Za-z0-9\-_]{30,}"          # PyPI upload token
    r"|//registry\.npmjs\.org/:_authToken=\S+"  # npm auth token
    # ── JWT / generic bearer ──────────────────────────────────────────────────
    r"|eyJ[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]+"  # JWT
    r"|Bearer\s+\S{20,}"
    # ── Private key material ──────────────────────────────────────────────────
    r"|-----BEGIN\s+(RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"
    # ── Database / connection strings ─────────────────────────────────────────
    r"|mongodb(\+srv)?://[^\"'\s]{10,}"
    r"|postgresql://[^\"'\s]{10,}"
    r"|mysql://[^\"'\s]{10,}"
    r"|redis://(:[^@\"'\s]+@)[^\"'\s]+"    # redis with password
    r"|DefaultEndpointsProtocol=https;AccountName=\S+"  # Azure storage
    # ── Env-var names that indicate credential material ───────────────────────
    r"|ANTHROPIC_API_KEY"
    r"|OPENAI_API_KEY"
    r"|OPENROUTER_API_KEY"
    r"|JIUWEN_API_KEY"
    r"|HUGGINGFACE_TOKEN"
    r"|HF_TOKEN"
    r"|GOOGLE_API_KEY"
    r"|STRIPE_API_KEY"
    r"|SENDGRID_API_KEY"
    r"|SLACK_BOT_TOKEN"
    r"|SLACK_USER_TOKEN"
    r"|GITHUB_TOKEN"
    r"|GITHUB_ACCESS_TOKEN"
    r"|AWS_SECRET_ACCESS_KEY"
    r"|AWS_SESSION_TOKEN"
    r"|DATABASE_URL"
    r"|DATABASE_PASSWORD"
    r"|REDIS_URL"
    r"|MONGO_URI"
    # ── Generic assignment patterns (value must follow immediately) ───────────
    r"|\bpassword\s*[=:]\s*\S+"
    r"|\bsecret\s*[=:]\s*\S+"
    r"|\bapi_key\s*[=:]\s*\S{8,}"
    r"|\btoken\s*[=:]\s*\S{10,}"
    r"|\bprivate_key\s*[=:]\s*\S{10,}"
    r")",
    re.IGNORECASE,
)

def _contains_secret(text: str) -> bool:
    return bool(SECRET_PATTERNS.search(text))
