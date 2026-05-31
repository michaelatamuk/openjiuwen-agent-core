# ══════════════════════════════════════════════════════════════════════════════
# BASELINE SKILL — deliberately shallow so GEPA has clear room to improve.
#
# What the baseline covers (generic, surface-level):
#   - Authentication presence
#   - Obvious vulnerabilities (SQL injection, missing validation)
#   - Sensitive data in responses
#
# What the baseline does NOT cover (hard examples will test this):
#   - JWT specifics (exp claim, algorithm confusion, weak secrets)
#   - SSRF via user-supplied URLs
#   - XXE in XML parsers
#   - OAuth CSRF (missing state parameter)
#   - Timing attacks / constant-time comparison
#   - ReDoS from catastrophic backtracking
#   - Mass assignment / client-controlled privilege escalation
#   - CORS misconfiguration details
#   - Rate limiting and brute-force protection
#   - PCI DSS / GDPR data minimisation rules
#   - Security headers (CSP, HSTS, X-Frame-Options)
#
# GEPA evolution should fill these gaps when trained on hard examples.
# ══════════════════════════════════════════════════════════════════════════════

SKILL_BODY = """\
# REST API Security Review

## Instructions

When reviewing an API endpoint or implementation for security issues:

- Check whether the endpoint requires authentication
- Look for obvious vulnerabilities such as SQL injection or missing input validation
- Check if sensitive data is exposed in responses or logs
- Verify that errors are handled safely without leaking internal details
- Suggest fixes for any issues found

Provide your findings in a clear, concise format.
"""
