# ══════════════════════════════════════════════════════════════════════════════
# EASY examples — issues any basic security checklist would catch.
# The baseline skill should handle most of these; they provide a performance
# floor that prevents evolution from regressing on obvious issues.
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "Review this Flask endpoint:\n\n"
        "from flask import Flask, jsonify\n\n"
        "app = Flask(__name__)\n\n"
        "@app.route('/api/admin/users')\n"
        "def list_users():\n"
        "    users = db.query('SELECT id, email FROM users')\n"
        "    return jsonify(users)"
    ),
    "expected_behavior": (
        "Must flag that the admin endpoint has no authentication or authorization check. "
        "Any unauthenticated request can list all user emails. "
        "Fix: add a `@login_required` decorator or verify the `Authorization` header "
        "with a valid Bearer token. "
        "Admin routes must also verify the caller has admin role or privilege — "
        "authentication alone is insufficient."
    ),
    "difficulty": "easy",
    "category": "authentication",
    "source": "golden",
}

example_02 = {
    "task_input": (
        "Review this API client call:\n\n"
        "import requests\n\n"
        "def fetch_data(user_id: str) -> dict:\n"
        "    url = f'https://api.service.com/data?api_key=sk-abc123&user={user_id}'\n"
        "    return requests.get(url).json()"
    ),
    "expected_behavior": (
        "Must flag that `api_key` is passed in the URL query string. "
        "Query parameters appear in server logs, browser history, proxy logs, and "
        "Referer headers — permanently exposing the secret. "
        "Fix: pass secrets in the `Authorization` header or as an `X-API-Key` header, "
        "never in the URL. "
        "Also flag that the API key is hardcoded in source code — "
        "use environment variables or a secrets manager instead."
    ),
    "difficulty": "easy",
    "category": "secrets",
    "source": "golden",
}

example_03 = {
    "task_input": (
        "Review this error handler:\n\n"
        "from flask import jsonify\n\n"
        "@app.errorhandler(Exception)\n"
        "def handle_error(e):\n"
        "    import traceback\n"
        "    return jsonify({\n"
        "        'error': str(e),\n"
        "        'traceback': traceback.format_exc()\n"
        "    }), 500"
    ),
    "expected_behavior": (
        "Must flag that the error handler returns the full stack trace to the client. "
        "This leaks internal file paths, library versions, database schema details, "
        "and business logic that attackers use for reconnaissance. "
        "Fix: log the full traceback server-side and return only a generic error message "
        "and a correlation ID to the client — `{'error': 'Internal server error', 'id': '...'}`. "
        "Never expose `traceback.format_exc()` or raw exception messages in API responses."
    ),
    "difficulty": "easy",
    "category": "information-disclosure",
    "source": "golden",
}

example_04 = {
    "task_input": (
        "Review this payment API documentation:\n\n"
        "# Payment API\n"
        "Base URL: http://api.payments.com/v1\n\n"
        "POST /charge\n"
        "Body: { 'amount': 100, 'card_number': '4111...', 'cvv': '123' }\n"
        "Response: { 'transaction_id': '...', 'status': 'success' }"
    ),
    "expected_behavior": (
        "Must flag that the payment API uses plain HTTP, not HTTPS. "
        "Card numbers and CVV are transmitted in cleartext — any network observer "
        "(ISP, proxy, WiFi attacker) can intercept them. "
        "This violates PCI DSS requirements. "
        "Fix: enforce HTTPS with HSTS header and require TLS 1.2+. "
        "Additionally, CVV must never be stored. "
        "Consider using a PCI-compliant tokenization service (Stripe, Braintree) "
        "instead of accepting raw card data."
    ),
    "difficulty": "easy",
    "category": "transport-security",
    "source": "golden",
}

GOLDEN_EXAMPLES_EASY = [example_01, example_02, example_03, example_04]
