"""
NagarAI -- best-effort citizen notifications.

Telegram only, not SMS or email: SMS needs a paid gateway and email needs SMTP
credentials, neither of which a hackathon budget covers, while the Telegram
Bot API is free and this project already runs a bot. Citizens who filed
through the web form have no push channel here -- they see status changes by
polling GET /api/mine (frontend/index.html already does this every few
seconds) and get a same-tab browser notification if they grant permission.
Say so rather than claiming a push channel that doesn't exist.

Reaching a citizen by chat_id requires keeping SOME contact information, which
narrows the privacy stance in docs/TECHNICAL_NOTE.md ("we count unique
reporters without holding identities"). See data/telegram_links.json
(gitignored): it maps the hashed citizen_id to a Telegram chat_id, written
only when that citizen messages the bot, and used only to deliver this one
notification -- never returned by any API response.

Every call here is best-effort: no Telegram token, no httpx, or a failed
request all degrade to a silent no-op rather than breaking the status update
that triggered it. A ward officer resolving an issue must never fail because
a notification couldn't be sent.
"""

import os

from backend import store

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def notify_resolved(cluster_row):
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        return
    try:
        import httpx
    except ImportError:
        return

    notified = set()
    for member in cluster_row.get("members", []):
        citizen_id = member.get("citizen_id")
        if not citizen_id or citizen_id in notified:
            continue
        notified.add(citizen_id)
        chat_id = store.get_telegram_chat(citizen_id)
        if not chat_id:
            continue
        try:
            httpx.post(TELEGRAM_API.format(token=token), json={
                "chat_id": chat_id,
                "text": (f"Update on your complaint — \"{cluster_row['description']}\" "
                         f"has been marked resolved. Reply with a photo if it "
                         f"still isn't fixed, or /start to check other complaints."),
            }, timeout=8.0)
        except Exception:
            pass
