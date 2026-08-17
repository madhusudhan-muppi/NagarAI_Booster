"""
NagarAI -- API server.

Run:  uvicorn backend.main:app --reload    (from the repository root)
Then: http://127.0.0.1:8000
"""

import os
import shutil
import sys
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import store                                    # noqa: E402
from backend.ml import intake                                 # noqa: E402
from backend.ml.dedup_engine import (                         # noqa: E402
    build_embedder, deduplicate, priority,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/clusters")
def get_clusters():
    rows, stats = build_queue()
    return {"clusters": rows, "stats": stats, "backends": intake.backend_status()}


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
):
    """Universal intake. Any combination of text, photo, and voice note."""
    if not text and photo is None and audio is None:
        raise HTTPException(400, "Send at least one of: text, photo, audio.")

    def save(upload):
        if upload is None:
            return None
        ext = os.path.splitext(upload.filename or "")[1] or ".bin"
        path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
        with open(path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        return path

    complaint, trace = intake.normalise(
        complaint_id=store.next_id(),
        citizen_id=citizen_id or f"u_{uuid.uuid4().hex[:4]}",
        text=text,
        audio_path=save(audio),
        image_path=save(photo),
        lat=lat, lon=lon, location_text=location_text,
    )
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
def update_status(key: str, payload: dict):
    status = payload.get("status")
    if status not in ("open", "in_progress", "resolved"):
        raise HTTPException(400, "status must be open, in_progress, or resolved")
    return store.set_status(key, status, payload.get("note", ""))


@app.post("/api/reset")
def reset():
    """Demo-day panic button: restore the judging set."""
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


@app.get("/")
def dashboard():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index):
        return JSONResponse({"error": "frontend/index.html not found"}, 404)
    return FileResponse(index)