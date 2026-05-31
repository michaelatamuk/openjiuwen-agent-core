# ══════════════════════════════════════════════════════════════════════════════
# MEDIUM examples — require specific security domain knowledge.
# The baseline skill will often miss these; evolved skills should catch them.
# Keywords: JWT, exp, iat, rate_limit, brute-force, CORS, IDOR, XSS,
#           mass_assignment, Content-Security-Policy, HSTS, PCI, CVV,
#           PAN, logging, data_minimization.
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "Review this JWT generation function:\n\n"
        "import jwt\n\n"
        "SECRET = 'my-secret-key'\n\n"
        "def generate_token(user_id: int) -> str:\n"
        "    payload = {'user_id': user_id, 'role': 'user'}\n"
        "    return jwt.encode(payload, SECRET, algorithm='HS256')"
    ),
    "expected_behavior": (
        "Must flag three issues: "
        "(1) No `exp` (expiration) claim — the JWT is valid forever; a stolen token "
        "never expires. Add `exp: datetime.utcnow() + timedelta(hours=1)`. "
        "(2) The secret `my-secret-key` is hardcoded and weak — use a cryptographically "
        "random 256-bit secret from an environment variable. "
        "(3) No `iat` (issued-at) or `jti` (JWT ID) claims, making token revocation "
        "impossible. Consider a short `exp` with a refresh token pattern."
    ),
    "difficulty": "medium",
    "category": "authentication",
    "source": "golden",
}

example_02 = {
    "task_input": (
        "Review this login endpoint:\n\n"
        "@app.route('/api/login', methods=['POST'])\n"
        "def login():\n"
        "    data = request.get_json()\n"
        "    user = User.query.filter_by(email=data['email']).first()\n"
        "    if user and user.check_password(data['password']):\n"
        "        return jsonify({'token': generate_token(user.id)})\n"
        "    return jsonify({'error': 'Invalid credentials'}), 401"
    ),
    "expected_behavior": (
        "Must flag: "
        "(1) No rate limiting — unlimited login attempts enable brute-force and "
        "credential stuffing attacks. Apply rate limiting per IP and per account "
        "with exponential backoff or account lockout after N failures. "
        "(2) No account lockout or CAPTCHA after repeated failures. "
        "(3) Verify the error message is identical for 'user not found' and "
        "'wrong password' to prevent user enumeration — the current message "
        "'Invalid credentials' is acceptable but must be consistent."
    ),
    "difficulty": "medium",
    "category": "authentication",
    "source": "golden",
}

example_03 = {
    "task_input": (
        "Review this CORS configuration:\n\n"
        "from flask_cors import CORS\n\n"
        "app = Flask(__name__)\n"
        "CORS(app, resources={r'/api/*': {'origins': '*'}})\n\n"
        "# JWT-authenticated API endpoints follow"
    ),
    "expected_behavior": (
        "Must flag the overly permissive CORS policy. `origins='*'` allows any domain "
        "to make cross-origin requests. For authenticated APIs this means malicious "
        "sites can trigger API calls on behalf of a victim who has a valid token. "
        "Fix: explicitly whitelist trusted origins. "
        "Additionally, `credentials=True` cannot be combined with `origins='*'` "
        "in browsers — CORS preflight will fail. "
        "Review also that the Access-Control-Allow-Methods and "
        "Access-Control-Allow-Headers are not overly permissive."
    ),
    "difficulty": "medium",
    "category": "cors",
    "source": "golden",
}

example_04 = {
    "task_input": (
        "Review this document download endpoint:\n\n"
        "@app.route('/api/documents/<int:doc_id>')\n"
        "@jwt_required\n"
        "def get_document(doc_id):\n"
        "    doc = Document.query.get_or_404(doc_id)\n"
        "    return send_file(doc.path)"
    ),
    "expected_behavior": (
        "Must identify the Insecure Direct Object Reference (IDOR) vulnerability. "
        "The endpoint checks authentication (`@jwt_required`) but does NOT verify "
        "that the authenticated user owns or has permission to access `doc_id`. "
        "Any authenticated user can access any document by guessing integer IDs. "
        "Fix: add an ownership check — `if doc.owner_id != current_user.id: abort(403)` "
        "— or enforce ownership in the query: "
        "`Document.query.filter_by(id=doc_id, owner_id=current_user.id).first_or_404()`."
    ),
    "difficulty": "medium",
    "category": "authorization",
    "source": "golden",
}

example_05 = {
    "task_input": (
        "Review this user profile update endpoint:\n\n"
        "@app.route('/api/profile', methods=['PUT'])\n"
        "@jwt_required\n"
        "def update_profile():\n"
        "    data = request.get_json()\n"
        "    current_user.bio = data.get('bio', '')\n"
        "    current_user.website = data.get('website', '')\n"
        "    db.session.commit()\n"
        "    return jsonify({'status': 'updated'})"
    ),
    "expected_behavior": (
        "Must flag: "
        "(1) No input validation — `bio` and `website` accept arbitrary content, "
        "enabling stored XSS if these values are rendered in a web frontend. "
        "(2) No length limits — unbounded `bio` enables storage DoS. "
        "(3) The `website` field must validate URL format and reject `javascript:` "
        "protocol URIs to prevent XSS via links. "
        "(4) Mass assignment risk — if the user model has additional fields like "
        "`is_admin` or `role`, accepting all `data` keys without an allowlist "
        "lets attackers escalate privileges."
    ),
    "difficulty": "medium",
    "category": "input-validation",
    "source": "golden",
}

example_06 = {
    "task_input": (
        "Review this Flask app configuration:\n\n"
        "from flask import Flask\n\n"
        "app = Flask(__name__)\n"
        "app.config['SECRET_KEY'] = 'dev-secret'\n"
        "app.config['DEBUG'] = True\n\n"
        "# Serves a public-facing API and web frontend"
    ),
    "expected_behavior": (
        "Must flag: "
        "(1) `DEBUG = True` in production enables the Werkzeug interactive debugger, "
        "which allows remote code execution via the debug console. "
        "(2) `SECRET_KEY = 'dev-secret'` is hardcoded and weak — use a random 256-bit "
        "key from an environment variable. "
        "(3) Missing security headers: `Content-Security-Policy`, "
        "`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, "
        "`Strict-Transport-Security` (HSTS), and `Referrer-Policy`. "
        "Use `flask-talisman` to apply them easily."
    ),
    "difficulty": "medium",
    "category": "configuration",
    "source": "golden",
}

example_07 = {
    "task_input": (
        "Review this user serializer:\n\n"
        "def serialize_user(user) -> dict:\n"
        "    return {\n"
        "        'id': user.id,\n"
        "        'email': user.email,\n"
        "        'ssn': user.ssn,\n"
        "        'dob': str(user.date_of_birth),\n"
        "        'credit_score': user.credit_score,\n"
        "        'password_hash': user.password_hash,\n"
        "    }"
    ),
    "expected_behavior": (
        "Must flag: "
        "(1) `password_hash` is returned — never expose password hashes in API responses; "
        "if the hash algorithm is weak, it enables offline cracking. "
        "(2) `ssn` (Social Security Number) is PII regulated by GDPR and CCPA; mask or omit it. "
        "(3) `credit_score` may be regulated financial data — verify caller authorization. "
        "(4) Apply the principle of data minimization — only return fields the caller "
        "needs. Use an explicit allowlist serializer, not a denylist approach."
    ),
    "difficulty": "medium",
    "category": "data-exposure",
    "source": "golden",
}

example_08 = {
    "task_input": (
        "Review this payment processor:\n\n"
        "import logging\n\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def process_payment(card_number: str, cvv: str, amount: float):\n"
        "    logger.info(f'Processing: card={card_number}, cvv={cvv}, amount={amount}')\n"
        "    result = payment_gateway.charge(card_number, cvv, amount)\n"
        "    logger.info(f'Result: {result}')\n"
        "    return result"
    ),
    "expected_behavior": (
        "Must flag: "
        "(1) Logging the full `card_number` (PAN) violates PCI DSS — only the last "
        "four digits may be logged. Log: `card=****{card_number[-4:]}`. "
        "(2) `cvv` must never be logged under any circumstances per PCI DSS. "
        "(3) The `result` object may contain sensitive card data — inspect before logging. "
        "This is a critical PCI DSS compliance violation that triggers card scheme fines "
        "and mandatory forensic audits."
    ),
    "difficulty": "medium",
    "category": "data-exposure",
    "source": "golden",
}

GOLDEN_EXAMPLES_MEDIUM = [
    example_01, example_02, example_03, example_04,
    example_05, example_06, example_07, example_08,
]
