"""
NagarAI -- zero-dependency server.

Identical API surface to backend/main.py, built entirely on the Python standard
library. No FastAPI, no uvicorn, no pydantic, no pip install of any kind.

Why this exists: a demo that cannot start because a venue's wifi dropped a
package download is a demo that scores zero. This file is the insurance policy.
Run backend/main.py when you have FastAPI; run this when you don't. The
dashboard and the Telegram bot talk to both without changes.

Run:  python3 backend/serve.py          (from the repository root)
Then: http://127.0.0.1:8000
"""

import json
import mimetypes
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend import env                                        # noqa: E402
env.load()

from backend import auth                                       # noqa: E402
from backend import notify                                     # noqa: E402
from backend import store                                     # noqa: E402
from backend.ml import intake                                  # noqa: E402
from backend.ml.dedup_engine import (                          # noqa: E402
    build_embedder, deduplicate, department_and_sla, parse_dt, priority,
)

DUPLICATE_WINDOW_HOURS = 24

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
FRONTEND = os.path.join(BASE_DIR, "frontend", "index.html")
GOV_FRONTEND = os.path.join(BASE_DIR, "frontend", "gov.html")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Minimal multipart/form-data parser
#
# The stdlib `cgi` module was removed in Python 3.13, so we parse it ourselves.
# Only what we need: text fields and file parts, one level, no nesting.
# ---------------------------------------------------------------------------

def parse_multipart(body, content_type):
    match = re.search(r"boundary=([^;]+)", content_type)
    if not match:
        return {}, {}
    boundary = match.group(1).strip('"').encode()
    fields, files = {}, {}

    for part in body.split(b"--" + boundary):
        if not part.strip() or part.strip() == b"--":
            continue
        head, _, content = part.partition(b"\r\n\r\n")
        if not content:
            continue
        content = content.rstrip(b"\r\n")
        header = head.decode("utf-8", "replace")

        name = re.search(r'name="([^"]*)"', header)
        if not name:
            continue
        name = name.group(1)

        filename = re.search(r'filename="([^"]*)"', header)
        if filename and filename.group(1):
            ext = os.path.splitext(filename.group(1))[1] or ".bin"
            path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
            with open(path, "wb") as f:
                f.write(content)
            files[name] = path
        else:
            fields[name] = content.decode("utf-8", "replace")

    return fields, files


# ---------------------------------------------------------------------------
# Pipeline -- identical logic to backend/main.py
# ---------------------------------------------------------------------------

def cluster_key(cluster):
    return min(c["id"] for c in cluster)


def find_duplicate_cluster(candidate, citizen_id, existing):
    """If `citizen_id` already has a complaint, filed within
    DUPLICATE_WINDOW_HOURS, that would land in the same cluster as `candidate`,
    return that cluster's key. Otherwise None.

    Runs the real dedup pipeline against a trial list (existing + candidate)
    rather than a cheaper heuristic, so "same issue" means exactly what it
    means everywhere else in this system -- the category/geo/semantic cascade,
    not a second, looser definition invented just for rate-limiting.
    """
    trial = existing + [candidate]
    vectors, backend, _ = build_embedder([c["description_en"] for c in trial])
    clusters, _ = deduplicate(trial, vectors, backend)
    mine = next((cl for cl in clusters if candidate in cl), None)
    if not mine or len(mine) < 2:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(hours=DUPLICATE_WINDOW_HOURS)
    for c in mine:
        if c is candidate or c.get("citizen_id") != citizen_id:
            continue
        if parse_dt(c["created_at"]) >= cutoff:
            return cluster_key(mine)
    return None


def build_queue():
    complaints = store.load_complaints()
    if not complaints:
        return [], {"complaints": 0, "clusters": 0, "merged": 0,
                    "embedding_backend": "n/a", "merge_log": []}

    vectors, backend, label = build_embedder(
        [c["description_en"] for c in complaints])
    clusters, merge_log = deduplicate(complaints, vectors, backend)
    statuses = store.load_statuses()

    rows = []
    for cl in clusters:
        p = priority(cl)
        key = cluster_key(cl)
        state = statuses.get(key, {})
        status = state.get("status", "open")
        sla = department_and_sla(
            cl, resolved_at=state.get("updated_at") if status == "resolved" else None)
        centre = next((c for c in cl if c.get("location_lat") is not None), None)
        cluster_merges = [m for m in merge_log if m["merged"][0] in {c["id"] for c in cl}]
        rows.append({
            "cluster_key": key,
            "category": p["category"],
            "hazard_lane": p["hazard_lane"],
            "description": cl[0]["description_en"],
            "location_text": cl[0].get("location_text"),
            "lat": centre["location_lat"] if centre else None,
            "lon": centre["location_lon"] if centre else None,
            "members": [
                {"id": c["id"], "raw_text": c["raw_text"],
                 "description_en": c["description_en"],
                 "secondary_category": c.get("secondary_category"),
                 "transcript": c.get("transcript"),
                 "severity": c["severity"], "photo_url": c.get("photo_url"),
                 "citizen_id": c.get("citizen_id")}
                for c in cl
            ],
            "affected_citizens": p["reporters"],
            "priority": p,
            "merges": cluster_merges,
            # Weakest link, not the average -- a cluster is only as trustworthy
            # as its shakiest merge. None for a cluster with one report: there
            # was no merge decision to be confident about.
            "merge_confidence": (min(m["confidence"] for m in cluster_merges)
                                 if cluster_merges else None),
            "has_conflict": any(c.get("secondary_category") for c in cl),
            "status": status,
            "status_note": state.get("note", ""),
            "updated_at": state.get("updated_at"),
            "verifications": store.load_verifications(key),
            **sla,
        })

    rows.sort(key=lambda r: (not r["hazard_lane"], -r["priority"]["final_score"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    return rows, {
        "complaints": len(complaints),
        "clusters": len(clusters),
        "merged": len(complaints) - len(clusters),
        "embedding_backend": label,
        "merge_log": merge_log,
    }


def my_complaints(citizen_id):
    """A citizen's own complaints, with the status of the cluster each landed in
    and, once resolved, their own latest "is it fixed?" response if any."""
    rows, _ = build_queue()
    mine = []
    for r in rows:
        for m in r["members"]:
            if m["citizen_id"] == citizen_id:
                my_verifications = [v for v in r["verifications"]
                                    if v["citizen_id"] == citizen_id]
                mine.append({
                    "id": m["id"], "description_en": m["description_en"],
                    "raw_text": m["raw_text"], "category": r["category"],
                    "status": r["status"], "status_note": r["status_note"],
                    "affected_citizens": r["affected_citizens"],
                    "cluster_key": r["cluster_key"],
                    "my_verification": my_verifications[-1] if my_verifications else None,
                })
    return mine


FORMULA_DOC = {
    "formula": "priority = S^2 x (1 + ln N) x (1 + D/7) x P",
    "terms": {
        "S": "severity 1-5, from the vision model -- never from citizen adjectives",
        "N": "unique reporters in the cluster, not complaint count",
        "D": "days pending, capped at 30",
        "P": "1.25 within 200 m of a school or hospital, else 1.0",
    },
    "lanes": "Hazard categories are worked ahead of the routine lane regardless of score.",
    "gaming_resistance": [
        "Duplicates merge, so filing 40 times yields one issue.",
        "N counts unique citizens, and ln damping makes 40 reporters 4.7x, not 40x.",
        "Severity comes from the image, so an angry rant cannot inflate it.",
        "Days pending is capped so nothing rises on age alone.",
    ],
}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "NagarAI/0.1"

    def handle_one_request(self):
        # A browser closing a connection mid-response is normal and not an error.
        # Left unhandled it prints a full traceback per reload, which during a
        # live demo looks like a crash.
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _send(self, payload, code=200):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _require_citizen(self):
        """Session dict, or None after sending 401 itself."""
        session = auth.require(self.headers, "citizen")
        if not session:
            self._send({"detail": "Sign in to do that."}, 401)
            return None
        return session

    def _require_gov(self):
        session = auth.require(self.headers, "gov")
        if not session:
            self._send({"detail": "Government sign-in required."}, 401)
            return None
        return session

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def _serve_file(self, path):
        if not os.path.exists(path):
            return self._send({"error": f"{os.path.basename(path)} not found"}, 404)
        body = open(path, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_binary(self, path):
        if not os.path.isfile(path) or os.path.dirname(path) != UPLOAD_DIR.rstrip("/"):
            return self._send({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        body = open(path, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            return self._serve_file(FRONTEND)

        if path in ("/gov", "/gov.html"):
            return self._serve_file(GOV_FRONTEND)

        if path.startswith("/uploads/"):
            # basename strips any directory components (including "../"), so
            # this can only ever resolve to a file directly inside UPLOAD_DIR.
            name = os.path.basename(path[len("/uploads/"):])
            return self._serve_binary(os.path.join(UPLOAD_DIR, name))

        if path == "/api/clusters":
            if not self._require_gov():
                return
            rows, stats = build_queue()
            return self._send({"clusters": rows, "stats": stats,
                               "backends": intake.backend_status()})

        if path == "/api/mine":
            session = self._require_citizen()
            if not session:
                return
            return self._send({"complaints": my_complaints(session["citizen_id"])})

        if path == "/api/geocode":
            if not self._require_citizen():
                return
            query = parse_qs(urlsplit(self.path).query).get("q", [""])[0]
            return self._send({"results": intake.geocode_search(query)})

        if path == "/api/formula":
            return self._send(FORMULA_DOC)

        if path == "/api/health":
            return self._send({"ok": True, "backends": intake.backend_status(),
                               "server": "stdlib"})

        self._send({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        ctype = self.headers.get("Content-Type", "")

        if path in ("/api/auth/signup", "/api/auth/login", "/api/auth/gov-login"):
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return self._send({"detail": "Malformed request body."}, 400)

            if path == "/api/auth/signup":
                token, error = auth.signup(payload.get("username"), payload.get("password"))
            elif path == "/api/auth/login":
                token, error = auth.login_citizen(payload.get("username"), payload.get("password"))
            else:
                token, error = auth.login_gov(payload.get("password"))

            if error:
                return self._send({"detail": error}, 400)
            return self._send({"token": token})

        if path == "/api/complaints":
            if ctype.startswith("multipart/form-data"):
                fields, files = parse_multipart(body, ctype)
            else:
                try:
                    fields, files = json.loads(body or b"{}"), {}
                except json.JSONDecodeError:
                    return self._send({"detail": "Malformed request body."}, 400)

            # Telegram already verifies identity (see bot/telegram_bot.py's hashed
            # user id), so the bot authenticates with a shared secret instead of a
            # citizen login and its declared citizen_id is trusted. Every other
            # caller must be a signed-in citizen -- citizen_id then comes from the
            # session, never the request body, so it cannot be spoofed to inflate
            # the priority formula's unique-reporter count.
            if auth.bot_secret_ok(self.headers):
                citizen_id = fields.get("citizen_id") or f"u_{uuid.uuid4().hex[:4]}"
            else:
                session = self._require_citizen()
                if not session:
                    return
                citizen_id = session["citizen_id"]

            text = (fields.get("text") or "").strip()
            if not text and not files:
                return self._send(
                    {"detail": "Send at least one of: text, photo, audio."}, 400)

            def num(key):
                try:
                    return float(fields[key])
                except (KeyError, TypeError, ValueError):
                    return None

            complaint, trace = intake.normalise(
                complaint_id=store.next_id(),
                citizen_id=citizen_id,
                text=text,
                audio_path=files.get("audio"),
                image_path=files.get("photo"),
                lat=num("lat"), lon=num("lon"),
                location_text=fields.get("location_text"),
            )

            existing = store.load_complaints()
            dup_key = find_duplicate_cluster(complaint, citizen_id, existing)
            if dup_key:
                return self._send({
                    "detail": (f"You already reported this issue in the last "
                              f"{DUPLICATE_WINDOW_HOURS}h. It's tracked as {dup_key}."),
                    "cluster_key": dup_key,
                }, 429)

            store.add_complaint(complaint)

            rows, _ = build_queue()
            joined = next((r for r in rows if complaint["id"]
                           in [m["id"] for m in r["members"]]), None)
            return self._send({
                "complaint": complaint,
                "trace": trace,
                "cluster_key": joined["cluster_key"] if joined else None,
                "merged_into_existing": bool(joined and len(joined["members"]) > 1),
                "affected_citizens": joined["affected_citizens"] if joined else 1,
            })

        match = re.match(r"^/api/clusters/([^/]+)/status$", path)
        if match:
            if not self._require_gov():
                return
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return self._send({"detail": "Malformed request body."}, 400)
            status = payload.get("status")
            if status not in ("open", "in_progress", "resolved"):
                return self._send(
                    {"detail": "status must be open, in_progress, or resolved"}, 400)
            key = match.group(1)
            result = store.set_status(key, status, payload.get("note", ""))
            if status == "resolved":
                rows, _ = build_queue()
                row = next((r for r in rows if r["cluster_key"] == key), None)
                if row:
                    notify.notify_resolved(row)
            return self._send(result)

        match = re.match(r"^/api/clusters/([^/]+)/verify$", path)
        if match:
            session = self._require_citizen()
            if not session:
                return
            if ctype.startswith("multipart/form-data"):
                fields, files = parse_multipart(body, ctype)
            else:
                try:
                    fields, files = json.loads(body or b"{}"), {}
                except json.JSONDecodeError:
                    return self._send({"detail": "Malformed request body."}, 400)

            key = match.group(1)
            rows, _ = build_queue()
            row = next((r for r in rows if r["cluster_key"] == key), None)
            if not row:
                return self._send({"detail": "No such cluster."}, 404)
            if session["citizen_id"] not in {m["citizen_id"] for m in row["members"]}:
                return self._send(
                    {"detail": "You did not file a complaint in this cluster."}, 403)

            record = {
                "citizen_id": session["citizen_id"],
                "confirmed": str(fields.get("confirmed", "")).lower() in ("true", "1", "yes"),
                "note": fields.get("note", ""),
                "photo_url": (os.path.basename(files["photo"])
                             if files.get("photo") else None),
                "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            return self._send({"verifications": store.add_verification(key, record)})

        if path == "/api/reset":
            if not self._require_gov():
                return
            return self._send({"restored": store.reset_to_seed()})

        self._send({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")


def main():
    port = int(os.environ.get("PORT", 8000))
    backends = intake.backend_status()
    print("NagarAI -- stdlib server (no external dependencies)")
    print(f"  embeddings : {backends['embeddings']}")
    print(f"  speech     : {backends['speech']}")
    print(f"  vision     : {backends['vision']}")
    print(f"  llm        : {backends.get('llm', 'n/a')}")
    if not os.environ.get("GOV_PASSWORD"):
        print("  WARNING: GOV_PASSWORD not set -- /gov login will always fail")
    print(f"\n  citizens : http://127.0.0.1:{port}")
    print(f"  officials: http://127.0.0.1:{port}/gov    Ctrl-C to stop\n")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()