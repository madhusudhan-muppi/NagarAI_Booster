"""
NagarAI -- API server.

Run:  uvicorn backend.main:app --reload    (from the repository root)
Then: http://127.0.0.1:8000
"""

import os
import shutil
import sys
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import env                                       # noqa: E402
env.load()

from backend import auth                                      # noqa: E402
from backend import notify                                    # noqa: E402
from backend import store                                    # noqa: E402
from backend.ml import intake                                 # noqa: E402
from backend.ml.dedup_engine import (                         # noqa: E402
    build_embedder, deduplicate, department_and_sla, parse_dt, priority,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DUPLICATE_WINDOW_HOURS = 24

app = FastAPI(title="NagarAI", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def cluster_key(cluster):
    """Stable identifier for a cluster: its lowest member id."""
    return min(c["id"] for c in cluster)


def find_duplicate_cluster(candidate, citizen_id, existing):
    """If `citizen_id` already has a complaint, filed within
    DUPLICATE_WINDOW_HOURS, that would land in the same cluster as `candidate`,
    return that cluster's key. Otherwise None. Runs the real dedup pipeline
    against a trial list rather than a separate heuristic, so "same issue"
    means the same thing here as everywhere else in the system."""
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
    """Complaints -> deduplicated, prioritised, status-annotated clusters."""
    complaints = store.load_complaints()
    if not complaints:
        return [], {"complaints": 0, "clusters": 0, "merged": 0}

    vectors, backend, backend_label = build_embedder(
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
            "merge_confidence": (min(m["confidence"] for m in cluster_merges)
                                 if cluster_merges else None),
            "has_conflict": any(c.get("secondary_category") for c in cl),
            "status": status,
            "status_note": state.get("note", ""),
            "updated_at": state.get("updated_at"),
            "verifications": store.load_verifications(key),
            **sla,
        })

    # Hazard lane first, then descending score inside each lane.
    rows.sort(key=lambda r: (not r["hazard_lane"], -r["priority"]["final_score"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    stats = {
        "complaints": len(complaints),
        "clusters": len(clusters),
        "merged": len(complaints) - len(clusters),
        "embedding_backend": backend_label,
        "merge_log": merge_log,
    }
    return rows, stats


# ---------------------------------------------------------------------------
# Auth
#
# Citizens: real accounts, so a complaint's citizen_id comes from a verified
# session rather than a client-supplied field -- see backend/auth.py.
# Telegram bot: exempt, authenticates with a shared secret (Telegram's own
# login is already the identity proof).
# Government: one shared password from GOV_PASSWORD gates the dashboard.
# ---------------------------------------------------------------------------

def _bearer(authorization):
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[7:]


def _require_citizen(authorization):
    session = auth.require_token(_bearer(authorization), "citizen")
    if not session:
        raise HTTPException(401, "Sign in to do that.")
    return session


def _require_gov(authorization):
    session = auth.require_token(_bearer(authorization), "gov")
    if not session:
        raise HTTPException(401, "Government sign-in required.")
    return session


@app.post("/api/auth/signup")
def signup(payload: dict):
    token, error = auth.signup(payload.get("username"), payload.get("password"))
    if error:
        raise HTTPException(400, error)
    return {"token": token}


@app.post("/api/auth/login")
def login(payload: dict):
    token, error = auth.login_citizen(payload.get("username"), payload.get("password"))
    if error:
        raise HTTPException(400, error)
    return {"token": token}


@app.post("/api/auth/gov-login")
def gov_login(payload: dict):
    token, error = auth.login_gov(payload.get("password"))
    if error:
        raise HTTPException(400, error)
    return {"token": token}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/clusters")
def get_clusters(authorization: str = Header(None)):
    _require_gov(authorization)
    rows, stats = build_queue()
    return {"clusters": rows, "stats": stats, "backends": intake.backend_status()}


@app.get("/api/mine")
def get_mine(authorization: str = Header(None)):
    session = _require_citizen(authorization)
    rows, _ = build_queue()
    mine = []
    for r in rows:
        for m in r["members"]:
            if m["citizen_id"] == session["citizen_id"]:
                my_verifications = [v for v in r["verifications"]
                                    if v["citizen_id"] == session["citizen_id"]]
                mine.append({
                    "id": m["id"], "description_en": m["description_en"],
                    "raw_text": m["raw_text"], "category": r["category"],
                    "status": r["status"], "status_note": r["status_note"],
                    "affected_citizens": r["affected_citizens"],
                    "cluster_key": r["cluster_key"],
                    "my_verification": my_verifications[-1] if my_verifications else None,
                })
    return {"complaints": mine}


@app.get("/api/geocode")
def geocode_lookup(q: str = "", authorization: str = Header(None)):
    """Live autocomplete for the citizen form's landmark field -- letting a
    citizen pick the right match is far more reliable than the single blind
    guess intake.geocode() falls back to server-side."""
    _require_citizen(authorization)
    return {"results": intake.geocode_search(q)}


@app.get("/api/formula")
def get_formula():
    """The priority model, served to the dashboard so it is never hidden."""
    return {
        "formula": "priority = S^2 x (1 + ln N) x (1 + D/7) x P",
        "terms": {
            "S": "severity 1-5, from the vision model -- never from citizen adjectives",
            "N": "unique reporters in the cluster, not complaint count",
            "D": "days pending, capped at 30",
            "P": "1.25 within 200 m of a school or hospital, else 1.0",
        },
        "lanes": "Hazard categories (live wire, gas leak, open manhole, wall "
                 "collapse) are worked ahead of the routine lane regardless of score.",
        "gaming_resistance": [
            "Duplicates merge, so filing 40 times yields one issue.",
            "N counts unique citizens, and ln damping makes 40 reporters 4.7x, not 40x.",
            "Severity comes from the image, so an angry rant cannot inflate it.",
            "Days pending is capped so nothing rises on age alone.",
        ],
    }


@app.post("/api/complaints")
async def create_complaint(
    text: str = Form(""),
    citizen_id: str = Form(None),
    lat: float = Form(None),
    lon: float = Form(None),
    location_text: str = Form(None),
    photo: UploadFile = File(None),
    audio: UploadFile = File(None),
    authorization: str = Header(None),
    x_bot_secret: str = Header(None),
):
    """Universal intake. Any combination of text, photo, and voice note."""
    if not text and photo is None and audio is None:
        raise HTTPException(400, "Send at least one of: text, photo, audio.")

    if auth.bot_secret_matches(x_bot_secret):
        citizen_id = citizen_id or f"u_{uuid.uuid4().hex[:4]}"
    else:
        citizen_id = _require_citizen(authorization)["citizen_id"]

    def save(upload):
        if upload is None:
            return None
        ext = os.path.splitext(upload.filename or "")[1] or ".bin"
        path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
        with open(path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        return path

    citizen_id = citizen_id or f"u_{uuid.uuid4().hex[:4]}"
    complaint, trace = intake.normalise(
        complaint_id=store.next_id(),
        citizen_id=citizen_id,
        text=text,
        audio_path=save(audio),
        image_path=save(photo),
        lat=lat, lon=lon, location_text=location_text,
    )

    existing = store.load_complaints()
    dup_key = find_duplicate_cluster(complaint, citizen_id, existing)
    if dup_key:
        return JSONResponse({
            "detail": (f"You already reported this issue in the last "
                      f"{DUPLICATE_WINDOW_HOURS}h. It's tracked as {dup_key}."),
            "cluster_key": dup_key,
        }, status_code=429)

    store.add_complaint(complaint)

    rows, _ = build_queue()
    joined = next((r for r in rows
                   if complaint["id"] in [m["id"] for m in r["members"]]), None)
    return {
        "complaint": complaint,
        "trace": trace,
        "cluster_key": joined["cluster_key"] if joined else None,
        "merged_into_existing": bool(joined and len(joined["members"]) > 1),
        "affected_citizens": joined["affected_citizens"] if joined else 1,
    }


@app.post("/api/clusters/{key}/status")
def update_status(key: str, payload: dict, authorization: str = Header(None)):
    _require_gov(authorization)
    status = payload.get("status")
    if status not in ("open", "in_progress", "resolved"):
        raise HTTPException(400, "status must be open, in_progress, or resolved")
    result = store.set_status(key, status, payload.get("note", ""))
    if status == "resolved":
        rows, _ = build_queue()
        row = next((r for r in rows if r["cluster_key"] == key), None)
        if row:
            notify.notify_resolved(row)
    return result


@app.post("/api/clusters/{key}/verify")
async def verify_cluster(
    key: str,
    confirmed: str = Form(""),
    note: str = Form(""),
    photo: UploadFile = File(None),
    authorization: str = Header(None),
):
    """A citizen's "is it fixed?" response to a resolved complaint of theirs."""
    session = _require_citizen(authorization)
    rows, _ = build_queue()
    row = next((r for r in rows if r["cluster_key"] == key), None)
    if not row:
        raise HTTPException(404, "No such cluster.")
    if session["citizen_id"] not in {m["citizen_id"] for m in row["members"]}:
        raise HTTPException(403, "You did not file a complaint in this cluster.")

    photo_name = None
    if photo is not None:
        photo_name = f"{uuid.uuid4().hex}{os.path.splitext(photo.filename or '')[1] or '.bin'}"
        with open(os.path.join(UPLOAD_DIR, photo_name), "wb") as f:
            shutil.copyfileobj(photo.file, f)

    record = {
        "citizen_id": session["citizen_id"],
        "confirmed": confirmed.lower() in ("true", "1", "yes"),
        "note": note,
        "photo_url": photo_name,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return {"verifications": store.add_verification(key, record)}


@app.post("/api/reset")
def reset(authorization: str = Header(None)):
    """Demo-day panic button: restore the judging set."""
    _require_gov(authorization)
    return {"restored": store.reset_to_seed()}


@app.get("/api/health")
def health():
    return {"ok": True, "backends": intake.backend_status()}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

if os.path.isdir(os.path.join(FRONTEND_DIR, "public")):
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "public")),
              name="static")

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
def citizen_page():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index):
        return JSONResponse({"error": "frontend/index.html not found"}, 404)
    return FileResponse(index)


@app.get("/gov")
def gov_dashboard():
    gov = os.path.join(FRONTEND_DIR, "gov.html")
    if not os.path.exists(gov):
        return JSONResponse({"error": "frontend/gov.html not found"}, 404)
    return FileResponse(gov)