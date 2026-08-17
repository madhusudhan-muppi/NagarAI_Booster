"""
NagarAI -- storage layer.

Deliberately a JSON file rather than Postgres. At municipal ward scale the
working set is thousands of complaints, not millions, and a hackathon demo that
depends on a database service is a demo that fails when the container does not
come up. Swapping this module for Postgres + pgvector is a contained change:
every caller goes through the functions below.
"""

import json
import os
import threading
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
COMPLAINTS_FILE = os.path.join(DATA_DIR, "complaints.json")
SEED_FILE = os.path.join(DATA_DIR, "seed_complaints.json")
STATUS_FILE = os.path.join(DATA_DIR, "cluster_status.json")

_lock = threading.Lock()


def _read(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)          # atomic -- a crash mid-write cannot truncate


def load_complaints():
    """Live complaints. Falls back to the seed set on first run."""
    with _lock:
        data = _read(COMPLAINTS_FILE, None)
        if data is None:
            data = _read(SEED_FILE, [])
            _write(COMPLAINTS_FILE, data)
        return data


def add_complaint(complaint):
    with _lock:
        data = _read(COMPLAINTS_FILE, None)
        if data is None:
            data = _read(SEED_FILE, [])
        data.append(complaint)
        _write(COMPLAINTS_FILE, data)
        return complaint


def next_id():
    data = load_complaints()
    n = len(data) + 1
    existing = {c["id"] for c in data}
    while f"C{n:03d}" in existing:
        n += 1
    return f"C{n:03d}"


def load_statuses():
    return _read(STATUS_FILE, {})


def set_status(cluster_key, status, note=""):
    with _lock:
        statuses = _read(STATUS_FILE, {})
        statuses[cluster_key] = {
            "status": status,
            "note": note,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _write(STATUS_FILE, statuses)
        return statuses[cluster_key]


def reset_to_seed():
    """Restore the judging set. Bind this to a demo-day panic button."""
    with _lock:
        seed = _read(SEED_FILE, [])
        _write(COMPLAINTS_FILE, seed)
        _write(STATUS_FILE, {})
        return len(seed)