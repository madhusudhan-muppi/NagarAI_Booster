"""
NagarAI -- authentication, stdlib only.

Citizens get real accounts (signup/login) so a filed complaint is tied to a
verified person. Before this module, POST /api/complaints trusted whatever
citizen_id the caller sent in the form -- a direct gaming vector against the
priority formula's unique-reporter count, since anyone could invent 40
citizen_ids for one issue. Now citizen_id comes from the session, never from
the request body.

The Telegram bot is exempt from citizen login: Telegram's own login is
already the identity proof (see bot/telegram_bot.py's hashed user id), so it
authenticates to this API with a shared secret instead of a password --
see bot_secret_ok() and BOT_SHARED_SECRET.

The government dashboard is not multi-tenant. One shared password from
GOV_PASSWORD gates it, which is a defensible scope for a ward console in a
hackathon demo without building a second account system.

Sessions are opaque tokens held in memory, not on disk. A server restart
logs everyone out -- an acceptable trade against shipping a session table.
"""

import hashlib
import hmac
import os
import secrets
import time

from backend import store

PBKDF2_ITERATIONS = 200_000


def _hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ITERATIONS).hex()
    return salt, digest


def _verify_password(password, salt, digest):
    _, check = _hash_password(password, salt)
    return hmac.compare_digest(check, digest)


_sessions = {}  # token -> {"role": "citizen" | "gov", ...claims}


def _issue(role, **claims):
    token = secrets.token_hex(24)
    _sessions[token] = {"role": role, "issued_at": time.time(), **claims}
    return token


def bearer_token(headers):
    value = headers.get("Authorization", "") or ""
    return value[7:] if value.startswith("Bearer ") else None


def require_token(token, role):
    """Session dict for a raw token string, or None. For callers that already
    extracted the token themselves (main.py's FastAPI Header dependency)."""
    if not token:
        return None
    session = _sessions.get(token)
    if not session or session["role"] != role:
        return None
    return session


def require(headers, role):
    """Session dict if `headers` (a dict-like with .get, e.g. BaseHTTPRequestHandler
    headers) carry a valid token for `role`, else None."""
    return require_token(bearer_token(headers), role)


def signup(username, password):
    username = (username or "").strip().lower()
    if not username or not password:
        return None, "username and password are required"
    if len(password) < 6:
        return None, "password must be at least 6 characters"
    citizen_id = f"u_{secrets.token_hex(4)}"
    salt, digest = _hash_password(password)
    ok = store.create_user(username, {
        "salt": salt, "digest": digest, "citizen_id": citizen_id,
        "created_at": time.time(),
    })
    if not ok:
        return None, "username already taken"
    return _issue("citizen", citizen_id=citizen_id, username=username), None


def login_citizen(username, password):
    username = (username or "").strip().lower()
    record = store.get_user(username)
    if not record or not _verify_password(password or "", record["salt"], record["digest"]):
        return None, "invalid username or password"
    return _issue("citizen", citizen_id=record["citizen_id"], username=username), None


def login_gov(password):
    expected = os.environ.get("GOV_PASSWORD")
    if not expected:
        return None, "GOV_PASSWORD is not configured on the server"
    if not hmac.compare_digest(password or "", expected):
        return None, "invalid password"
    return _issue("gov"), None


def bot_secret_matches(secret):
    expected = os.environ.get("BOT_SHARED_SECRET")
    if not expected:
        return False
    return hmac.compare_digest(secret or "", expected)


def bot_secret_ok(headers):
    return bot_secret_matches(headers.get("X-Bot-Secret", ""))
