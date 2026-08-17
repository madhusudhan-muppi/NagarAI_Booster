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
import os
import re
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend import store                                     # noqa: E402
from backend.ml import intake                                  # noqa: E402
from backend.ml.dedup_engine import (                          # noqa: E402
    build_embedder, deduplicate, priority,
)

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
FRONTEND = os.path.join(BASE_DIR, "frontend", "index.html")
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
        centre = next((c for c in cl if c.get("location_lat") is not None), None)
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
                "transcript": c.get("transcript"),
                "severity": c["severity"], "photo_url": c.get("photo_url")}
                for c in cl
            ],
            "affected_citizens": p["reporters"],
            "priority": p,
            "status": state.get("status", "open"),
            "status_note": state.get("note", ""),
            "updated_at": state.get("updated_at"),
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

    def _send(self, payload, code=200):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            if not os.path.exists(FRONTEND):
                return self._send({"error": "frontend/index.html not found"}, 404)
            body = open(FRONTEND, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/clusters":
            rows, stats = build_queue()
            return self._send({"clusters": rows, "stats": stats,
                               "backends": intake.backend_status()})

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

        if path == "/api/complaints":
            if ctype.startswith("multipart/form-data"):
                fields, files = parse_multipart(body, ctype)
            else:
                try:
                    fields, files = json.loads(body or b"{}"), {}
                except json.JSONDecodeError:
                    return self._send({"detail": "Malformed request body."}, 400)

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
                citizen_id=fields.get("citizen_id") or f"u_{uuid.uuid4().hex[:4]}",
                text=text,
                audio_path=files.get("audio"),
                image_path=files.get("photo"),
                lat=num("lat"), lon=num("lon"),
                location_text=fields.get("location_text"),
            )
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
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                return self._send({"detail": "Malformed request body."}, 400)
            status = payload.get("status")
            if status not in ("open", "in_progress", "resolved"):
                return self._send(
                    {"detail": "status must be open, in_progress, or resolved"}, 400)
            return self._send(store.set_status(match.group(1), status,
                                               payload.get("note", "")))

        if path == "/api/reset":
            return self._send({"restored": store.reset_to_seed()})

        self._send({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")
        
    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def main():
    port = int(os.environ.get("PORT", 8000))
    backends = intake.backend_status()
    print("NagarAI -- stdlib server (no external dependencies)")
    print(f"  embeddings : {backends['embeddings']}")
    print(f"  speech     : {backends['speech']}")
    print(f"  vision     : {backends['vision']}")
    print(f"\n  http://127.0.0.1:{port}    Ctrl-C to stop\n")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()