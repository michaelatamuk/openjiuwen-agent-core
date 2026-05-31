# ══════════════════════════════════════════════════════════════════════════════
# HARD examples — require deep security expertise; the baseline skill will
# almost always miss these entirely.  Technical keywords that only an evolved
# skill should produce:
#   alg_confusion, RS256, HS256, SSRF, 169.254.169.254, OAuth, state_parameter,
#   CSRF, PKCE, mass_assignment, setattr, XXE, defusedxml, no_network,
#   resolve_entities, hmac.compare_digest, constant_time, ReDoS, backtracking,
#   privilege_escalation, client-controlled, role_parameter.
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "Review this JWT verification function:\n\n"
        "import jwt\n\n"
        "PUBLIC_KEY = open('public.pem').read()\n\n"
        "def verify_token(token: str) -> dict:\n"
        "    header = jwt.get_unverified_header(token)\n"
        "    algorithm = header.get('alg', 'RS256')\n"
        "    return jwt.decode(token, PUBLIC_KEY, algorithms=[algorithm])"
    ),
    "expected_behavior": (
        "Must identify the JWT algorithm confusion (alg_confusion) attack. "
        "The code trusts the `alg` field from the unverified token header. "
        "An attacker changes `alg` to `HS256` and signs the token with the PUBLIC_KEY "
        "as the HMAC secret — since the public key is public, anyone can forge tokens. "
        "The server verifies the HS256 HMAC using PUBLIC_KEY as secret, which succeeds. "
        "Fix: never read `alg` from the token; hardcode the expected algorithm: "
        "`jwt.decode(token, PUBLIC_KEY, algorithms=['RS256'])`. "
        "This is CVE-2015-9235 class vulnerability."
    ),
    "difficulty": "hard",
    "category": "authentication",
    "source": "golden",
}

example_02 = {
    "task_input": (
        "Review this webhook registration endpoint:\n\n"
        "@app.route('/api/webhooks', methods=['POST'])\n"
        "@jwt_required\n"
        "def register_webhook():\n"
        "    url = request.json['callback_url']\n"
        "    # Verify the webhook works before saving\n"
        "    test_response = requests.get(url, timeout=5)\n"
        "    if test_response.status_code == 200:\n"
        "        Webhook.create(user_id=current_user.id, url=url)\n"
        "        return jsonify({'status': 'registered'})"
    ),
    "expected_behavior": (
        "Must identify the Server-Side Request Forgery (SSRF) vulnerability. "
        "The server fetches an attacker-controlled URL without restriction, allowing: "
        "(1) Access to cloud instance metadata: `http://169.254.169.254/latest/meta-data/` "
        "exposes AWS IAM credentials. "
        "(2) Port scanning of internal networks by observing timeouts vs connections. "
        "(3) Access to internal services: Redis on `http://localhost:6379`, "
        "internal admin panels, databases. "
        "Fix: validate the URL against an allowlist of permitted domains; "
        "block RFC-1918 private ranges, loopback (127.x), and cloud metadata IPs "
        "with a DNS rebinding-resistant HTTP client. "
        "This is OWASP Top 10 A10 — SSRF."
    ),
    "difficulty": "hard",
    "category": "ssrf",
    "source": "golden",
}

example_03 = {
    "task_input": (
        "Review this OAuth2 callback handler:\n\n"
        "@app.route('/auth/callback')\n"
        "def oauth_callback():\n"
        "    code = request.args.get('code')\n"
        "    token = oauth_client.exchange_code(code)\n"
        "    user_info = oauth_client.get_user(token)\n"
        "    user = User.find_or_create(email=user_info['email'])\n"
        "    login_user(user)\n"
        "    return redirect('/')"
    ),
    "expected_behavior": (
        "Must flag: "
        "(1) Missing `state` parameter validation. OAuth2 requires a random `state` "
        "value generated before the authorization redirect, stored in the session, "
        "and verified in the callback to prevent OAuth CSRF. Without it, an attacker "
        "tricks the victim into completing the OAuth flow that logs them into the "
        "attacker's account (account fixation / CSRF attack). "
        "Fix: `state = secrets.token_urlsafe()`, store in session, "
        "verify: `if request.args['state'] != session['oauth_state']: abort(400)`. "
        "(2) Missing PKCE for public clients. "
        "(3) No validation that the `code` was issued to this client."
    ),
    "difficulty": "hard",
    "category": "oauth",
    "source": "golden",
}

example_04 = {
    "task_input": (
        "Review this account update endpoint:\n\n"
        "@app.route('/api/account', methods=['PATCH'])\n"
        "@jwt_required\n"
        "def update_account():\n"
        "    data = request.get_json()\n"
        "    for key, value in data.items():\n"
        "        setattr(current_user, key, value)\n"
        "    db.session.commit()\n"
        "    return jsonify(current_user.to_dict())"
    ),
    "expected_behavior": (
        "Must identify the mass assignment vulnerability. "
        "The endpoint calls `setattr` on every key from the client payload without "
        "an allowlist, so an attacker can POST `{'is_admin': true, 'role': 'admin'}` "
        "to escalate privileges or `{'email': 'attacker@evil.com'}` to hijack the account. "
        "Fix: use an explicit allowlist of mutable fields: "
        "`ALLOWED = {'name', 'bio', 'phone'}; "
        "[setattr(current_user, k, data[k]) for k in ALLOWED if k in data]`. "
        "Never use `setattr` with unvalidated client-supplied keys on database models. "
        "Use a schema validation library (Marshmallow, Pydantic) with explicit fields."
    ),
    "difficulty": "hard",
    "category": "authorization",
    "source": "golden",
}

example_05 = {
    "task_input": (
        "Review this XML import endpoint:\n\n"
        "from lxml import etree\n\n"
        "@app.route('/api/import', methods=['POST'])\n"
        "def import_data():\n"
        "    xml_data = request.data\n"
        "    parser = etree.XMLParser()\n"
        "    root = etree.fromstring(xml_data, parser)\n"
        "    items = [elem.text for elem in root.findall('item')]\n"
        "    return jsonify({'imported': len(items)})"
    ),
    "expected_behavior": (
        "Must identify the XML External Entity (XXE) injection vulnerability. "
        "The default `lxml.etree.XMLParser()` processes external entity declarations. "
        "An attacker submits XML with `<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>` "
        "to read local files, or uses HTTP URIs to trigger SSRF. "
        "Fix: disable external entities explicitly: "
        "`etree.XMLParser(no_network=True, resolve_entities=False, load_dtd=False)`. "
        "Alternatively use the `defusedxml` library which disables all dangerous "
        "XML features by default. "
        "This is OWASP Top 10 A05 — XXE injection."
    ),
    "difficulty": "hard",
    "category": "injection",
    "source": "golden",
}

example_06 = {
    "task_input": (
        "Review this webhook signature verification:\n\n"
        "import hmac, hashlib\n\n"
        "def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:\n"
        "    expected = hmac.new(\n"
        "        secret.encode(), payload, hashlib.sha256\n"
        "    ).hexdigest()\n"
        "    return signature == expected"
    ),
    "expected_behavior": (
        "Must identify the timing attack vulnerability. "
        "The `==` string comparison short-circuits — it returns False as soon as "
        "the first differing character is found. An attacker measures response times "
        "to learn how many leading characters of a forged signature are correct, "
        "enabling byte-by-byte brute-force of the HMAC. "
        "Fix: replace `signature == expected` with `hmac.compare_digest(signature, expected)` "
        "which performs a constant-time comparison regardless of where strings differ. "
        "This is the single required change; `hmac.compare_digest` is the canonical "
        "Python solution for constant-time secret comparison."
    ),
    "difficulty": "hard",
    "category": "cryptography",
    "source": "golden",
}

example_07 = {
    "task_input": (
        "Review this input validation function used in an API:\n\n"
        "import re\n\n"
        "EMAIL_RE = re.compile(\n"
        "    r'^([a-zA-Z0-9]+[._-]?)*[a-zA-Z0-9]+'\n"
        "    r'@([a-zA-Z0-9]+[._-]?)*[a-zA-Z0-9]+\\.[a-zA-Z]{2,}$'\n"
        ")\n\n"
        "def validate_email(email: str) -> bool:\n"
        "    return bool(EMAIL_RE.match(email))"
    ),
    "expected_behavior": (
        "Must identify the ReDoS (Regular Expression Denial of Service) vulnerability. "
        "The pattern contains nested quantifiers `([a-zA-Z0-9]+[._-]?)*` — "
        "the outer `*` and inner `+` cause catastrophic backtracking on crafted inputs "
        "like `aaaaaaaaaaaaaaaaaaaaaaaaaaaa@` (long prefix with no `@` match). "
        "A single API request with such input can block the event loop for seconds. "
        "Fix: use a linear-time regex without nested quantifiers, or set a strict "
        "length limit (`if len(email) > 254: return False`) before matching, "
        "or use the `email-validator` library which uses a safe implementation. "
        "ReDoS is a common API denial-of-service vector."
    ),
    "difficulty": "hard",
    "category": "denial-of-service",
    "source": "golden",
}

example_08 = {
    "task_input": (
        "Review this search endpoint:\n\n"
        "@app.route('/api/search')\n"
        "def search():\n"
        "    query = request.args.get('q', '')\n"
        "    role  = request.args.get('role', 'user')\n"
        "    if role == 'admin':\n"
        "        results = db.search_all(query)\n"
        "    else:\n"
        "        results = db.search_public(query)\n"
        "    return jsonify(results)"
    ),
    "expected_behavior": (
        "Must identify the client-controlled privilege escalation vulnerability. "
        "The `role` parameter is taken directly from the query string, so any caller "
        "can pass `?role=admin` to access all records. "
        "Authorization decisions must never come from client-supplied parameters. "
        "Fix: read the role from the authenticated session or JWT claims: "
        "`role = current_user.role` (after `@jwt_required`). "
        "Additionally: no authentication check at all — `@jwt_required` is missing. "
        "This is a Broken Access Control vulnerability (OWASP Top 10 A01) via "
        "role parameter tampering."
    ),
    "difficulty": "hard",
    "category": "authorization",
    "source": "golden",
}

GOLDEN_EXAMPLES_HARD = [
    example_01, example_02, example_03, example_04,
    example_05, example_06, example_07, example_08,
]
