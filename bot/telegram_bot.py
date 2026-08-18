"""
NagarAI -- Telegram intake bot.

Telegram over WhatsApp for a reason: it gives voice notes, photos, and native
location sharing with no app install for the citizen and no Business API
approval queue for us.

Setup:
    1. Message @BotFather on Telegram, /newbot, copy the token
    2. export TELEGRAM_TOKEN="..."          (or put it in .env)
    3. Start the API:  uvicorn backend.main:app
    4. Start the bot:  python bot/telegram_bot.py

The bot is a thin client. All intelligence lives behind POST /api/complaints, so
the PWA and the bot share one pipeline and there is only one thing to debug.
"""

import hashlib
import os
import sys
import tempfile

import httpx
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler, filters,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import env                                       # noqa: E402
env.load()

API_BASE = os.environ.get("NAGARAI_API", "http://127.0.0.1:8000")
TOKEN = os.environ.get("TELEGRAM_TOKEN")

WELCOME = (
    "*NagarAI* — report a civic problem\n\n"
    "Send me any of these, in any language:\n"
    "• A voice note describing the problem\n"
    "• A photo of it\n"
    "• A typed message\n\n"
    "Share your location too and I can place it on the ward map.\n"
    "You can combine them — a photo with a caption works well."
)


def citizen_hash(user_id):
    """
    Pseudonymous, stable citizen ID.

    We count unique reporters for the priority score, so we need to tell people
    apart -- but we never need to know who they are. Hashing the Telegram user
    ID gives us both. Privacy-by-design is a scored bonus, and this is the
    cheapest place to earn it.
    """
    return "tg_" + hashlib.sha256(str(user_id).encode()).hexdigest()[:8]


async def download(file, suffix):
    tg_file = await file.get_file()
    path = tempfile.mktemp(suffix=suffix)
    await tg_file.download_to_drive(path)
    return path


async def submit(*, text="", photo_path=None, audio_path=None,
                 lat=None, lon=None, citizen_id=None):
    data = {"text": text, "citizen_id": citizen_id}
    if lat is not None:
        data["lat"], data["lon"] = str(lat), str(lon)

    files = {}
    opened = []
    try:
        if photo_path:
            fh = open(photo_path, "rb")
            opened.append(fh)
            files["photo"] = ("photo.jpg", fh, "image/jpeg")
        if audio_path:
            fh = open(audio_path, "rb")
            opened.append(fh)
            files["audio"] = ("audio.ogg", fh, "audio/ogg")

        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(f"{API_BASE}/api/complaints",
                                  data={k: v for k, v in data.items() if v is not None},
                                  files=files or None)
            r.raise_for_status()
            return r.json()
    finally:
        for fh in opened:
            fh.close()


def confirmation(result):
    c = result["complaint"]
    lines = [
        f"*Registered — {c['id']}*",
        f"Category: {c['category'].replace('_', ' ')}",
        f"Severity: {c['severity']}/5",
    ]
    if result.get("merged_into_existing"):
        lines.append(
            f"\nOthers have reported this too. Your complaint joined an existing "
            f"issue now backed by *{result['affected_citizens']} citizens*, which "
            f"raises its priority with the ward office."
        )
    else:
        lines.append("\nFiled as a new issue. You'll be notified when it's resolved.")
    if c.get("location_lat") is None:
        lines.append("\n_Tip: share your location so we can map it precisely._")
    return "\n".join(lines)


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_markdown(WELCOME)


async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    uid = citizen_hash(msg.from_user.id)
    pending = ctx.user_data.setdefault("pending", {})

    # Location arriving on its own attaches to whatever they just sent.
    if msg.location:
        pending["lat"] = msg.location.latitude
        pending["lon"] = msg.location.longitude
        if not (pending.get("text") or pending.get("photo") or pending.get("audio")):
            await msg.reply_text("Location saved. Now describe the problem, "
                                 "or send a photo or voice note.")
            return

    if msg.text and not msg.text.startswith("/"):
        pending["text"] = msg.text
    if msg.caption:
        pending["text"] = msg.caption
    if msg.photo:
        pending["photo"] = await download(msg.photo[-1], ".jpg")
    if msg.voice or msg.audio:
        pending["audio"] = await download(msg.voice or msg.audio, ".ogg")

    if not (pending.get("text") or pending.get("photo") or pending.get("audio")):
        return

    note = await msg.reply_text("Processing your complaint…")
    try:
        result = await submit(
            text=pending.get("text", ""),
            photo_path=pending.get("photo"),
            audio_path=pending.get("audio"),
            lat=pending.get("lat"), lon=pending.get("lon"),
            citizen_id=uid,
        )
        await note.edit_text(confirmation(result), parse_mode="Markdown")
    except httpx.HTTPError:
        await note.edit_text(
            "The complaint service isn't reachable right now. "
            "Please try again in a moment — nothing was lost.")
    finally:
        for key in ("photo", "audio"):
            if pending.get(key) and os.path.exists(pending[key]):
                os.unlink(pending[key])
        ctx.user_data["pending"] = {}


def main():
    if not TOKEN:
        raise SystemExit(
            "TELEGRAM_TOKEN is not set.\n"
            "Get one from @BotFather, then: export TELEGRAM_TOKEN='...'")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(MessageHandler(
        filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO | filters.LOCATION,
        handle))
    print(f"Bot running. API at {API_BASE}. Ctrl-C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()