"""
NagarAI — Deduplication and Explainable Priority Engine
=======================================================

Merges complaints describing the same civic issue into one cluster, then ranks
clusters with a transparent, gaming-resistant priority formula.

Design decisions (be ready to defend these in Q&A):

1. NOT k-means. We do not know the number of distinct issues in advance, and
   complaints arrive as a stream. We use incremental greedy clustering with a
   union-find structure, so a new complaint is compared only against existing
   cluster members and merges in O(n) per arrival.

2. Three-stage matching, cheapest filter first:
      category gate  ->  geographic gate  ->  semantic similarity
   A pothole never merges with a garbage complaint no matter how similar the
   wording, and two identical descriptions 4 km apart stay separate. This is
   what stops the classic embedding-only failure where "pothole on main road"
   in two different wards collapses into one issue.

3. Per-category merge radii. Waterlogging genuinely spreads across a stretch of
   road; a pothole is a point defect. One global radius is wrong for both.

4. Missing GPS is handled, not dropped. If either complaint lacks coordinates we
   fall back to location-text overlap with a raised similarity threshold.

Run:  python dedup_engine.py ../../data/seed_complaints.json
"""

import json
import math
import re
import sys
from collections import defaultdict

# ----------------------------------------------------------------------------
# Tunable parameters -- tune these against your own adversarial test set
# ----------------------------------------------------------------------------

# Maximum distance in metres for two complaints of a category to be the same issue
MERGE_RADIUS_M = {
    "pothole": 100,
    "garbage": 60,
    "streetlight": 120,
    "waterlogging": 200,   # floods spread along a stretch
    "live_wire": 80,
    "other": 100,
}
DEFAULT_RADIUS_M = 100

# Text similarity floors. These are backend-dependent: sentence-transformer
# cosines sit much higher than TF-IDF cosines for the same pair, so the
# threshold is selected from the embedder actually in use.
SIM_THRESHOLD = {
    "sbert": 0.42,
    "tfidf": 0.15,
}
SIM_THRESHOLD_NO_GPS = {   # raised floor when we fall back to text location
    "sbert": 0.55,
    "tfidf": 0.20,
}

# If two same-category complaints are within this fraction of the merge radius
# they are treated as the same physical spot and merged on geography alone.
# Two garbage reports 13 m apart are the same bin whatever words were used --
# demanding text agreement there just loses recall on terse or mistyped reports.
PROXIMITY_DOMINANT_FRACTION = 0.35

# When both complaints carry a CLIP image embedding we blend it with the text
# similarity. Text still leads: a photo tells you the KIND of defect reliably
# but two different potholes photograph almost identically, so image similarity
# alone would over-merge. It earns its place on terse or badly worded reports
# where the text signal is thin.
TEXT_WEIGHT = 0.6
IMAGE_WEIGHT = 0.4

# Categories that jump the queue regardless of how many people complained
HAZARD_CATEGORIES = {"live_wire", "gas_leak", "open_manhole", "wall_collapse"}

# Sensitive locations that raise priority (lat, lon, label)
SENSITIVE_SITES = [
    (12.96940, 80.23880, "Government Higher Secondary School, Velachery"),
    (13.00810, 80.21290, "Guindy Government Hospital"),
]
SENSITIVE_RADIUS_M = 200
PROXIMITY_MULTIPLIER = 1.25


# ----------------------------------------------------------------------------
# Geography
# ----------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ----------------------------------------------------------------------------
# Text embedding
#
# Tier 1: sentence-transformers (what we ship on demo day).
# Tier 2: a dependency-free TF-IDF fallback so this script runs on any laptop
#         with no model download -- useful when venue wifi dies.
# ----------------------------------------------------------------------------

_ST_MODEL = None


def build_embedder(texts):
    global _ST_MODEL
    try:
        if _ST_MODEL is None:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        model = _ST_MODEL
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors], "sbert", "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        return _tfidf(texts), "tfidf", "TF-IDF fallback (no ML model downloaded)"


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of",
    "for", "and", "or", "it", "this", "that", "there", "near", "beside", "with",
    "has", "have", "been", "be", "very", "not", "no", "after", "our", "my",
}


def _tokenize(text):
    return [t for t in re.findall(r"[a-z]+", text.lower())
            if t not in STOPWORDS and len(t) > 2]


def _tfidf(texts):
    docs = [_tokenize(t) for t in texts]
    df = defaultdict(int)
    for d in docs:
        for term in set(d):
            df[term] += 1
    n = len(docs)
    vocab = sorted(df)
    idx = {t: i for i, t in enumerate(vocab)}

    vectors = []
    for d in docs:
        tf = defaultdict(int)
        for term in d:
            tf[term] += 1
        vec = [0.0] * len(vocab)
        for term, count in tf.items():
            vec[idx[term]] = (count / max(len(d), 1)) * math.log((1 + n) / (1 + df[term]) + 1)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


# ----------------------------------------------------------------------------
# Union-find
# ----------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)
            return True
        return False


# ----------------------------------------------------------------------------
# Deduplication
# ----------------------------------------------------------------------------

def deduplicate(complaints, vectors, backend="sbert"):
    """Return (clusters, merge_log). Every merge records why it happened."""
    uf = UnionFind(len(complaints))
    merge_log = []
    sim_floor = SIM_THRESHOLD[backend]
    sim_floor_nogps = SIM_THRESHOLD_NO_GPS[backend]

    for i in range(len(complaints)):
        for j in range(i + 1, len(complaints)):
            a, b = complaints[i], complaints[j]

            # Stage 1 -- category gate
            if a["category"] != b["category"]:
                continue

            text_sim = cosine(vectors[i], vectors[j])
            va, vb = a.get("image_vec"), b.get("image_vec")
            if va and vb and len(va) == len(vb):
                image_sim = cosine(va, vb)
                sim = TEXT_WEIGHT * text_sim + IMAGE_WEIGHT * image_sim
                sim_note = f"text {text_sim:.2f} + image {image_sim:.2f}"
            else:
                sim = text_sim
                image_sim = None
                sim_note = f"text {text_sim:.2f}"

            has_gps = (a.get("location_lat") is not None and
                       b.get("location_lat") is not None)

            # Stage 2 -- geographic gate
            if has_gps:
                dist = haversine_m(a["location_lat"], a["location_lon"],
                                   b["location_lat"], b["location_lon"])
                radius = MERGE_RADIUS_M.get(a["category"], DEFAULT_RADIUS_M)
                if dist > radius:
                    continue

                # Stage 3a -- proximity dominant: same spot, same category
                if dist <= PROXIMITY_DOMINANT_FRACTION * radius:
                    rule = "proximity-dominant"
                    reason = (f"{dist:.0f} m apart, within "
                              f"{PROXIMITY_DOMINANT_FRACTION:.0%} of the "
                              f"{radius} m radius ({sim_note})")
                else:
                    # Stage 3b -- borderline distance, text must agree
                    if sim < sim_floor:
                        continue
                    rule = "geo + semantic"
                    reason = (f"{dist:.0f} m apart (limit {radius} m), "
                              f"{sim_note} = {sim:.2f} >= {sim_floor}")
            else:
                # Fallback -- no coordinates, so require location-text overlap
                ta = set(_tokenize(a.get("location_text") or ""))
                tb = set(_tokenize(b.get("location_text") or ""))
                if not ta or not tb:
                    continue
                overlap = len(ta & tb) / len(ta | tb)
                if overlap < 0.30 or sim < sim_floor_nogps:
                    continue
                rule = "no-GPS text fallback"
                reason = (f"no GPS; location-text overlap {overlap:.2f}, "
                          f"{sim_note} = {sim:.2f} >= {sim_floor_nogps}")

            if uf.union(i, j):
                merge_log.append({
                    "merged": (a["id"], b["id"]),
                    "similarity": round(sim, 3),
                    "image_similarity": (round(image_sim, 3)
                                         if image_sim is not None else None),
                    "rule": rule,
                    "reason": reason,
                    "category": a["category"],
                })

    groups = defaultdict(list)
    for i in range(len(complaints)):
        groups[uf.find(i)].append(complaints[i])
    return list(groups.values()), merge_log


# ----------------------------------------------------------------------------
# Explainable priority
#
#   priority = S^2  x  (1 + ln N)  x  (1 + D/7)  x  P
#
#   S = max severity in the cluster (from the vision model, never from the
#       citizen's adjectives -- an angry rant must not inflate severity)
#   N = unique reporters (not complaint count -- a forty-person WhatsApp
#       brigade is still one issue with N unique citizens)
#   D = days pending, capped so nothing floats up purely by ageing
#   P = 1.25 within 200 m of a school or hospital, else 1.0
#
#   Severity is squared and headcount is logarithmic on purpose: a 40-report
#   pothole must not outrank a 2-report live wire. Hazard categories are
#   additionally placed in a separate lane that always sorts first.
# ----------------------------------------------------------------------------

DAYS_CAP = 30


def priority(cluster):
    severity = max(c["severity"] for c in cluster)
    reporters = len({c["citizen_id"] for c in cluster})
    days = min(max(c["days_pending"] for c in cluster), DAYS_CAP)
    category = cluster[0]["category"]

    proximity, site = 1.0, None
    for c in cluster:
        if c.get("location_lat") is None:
            continue
        for lat, lon, label in SENSITIVE_SITES:
            if haversine_m(c["location_lat"], c["location_lon"], lat, lon) <= SENSITIVE_RADIUS_M:
                proximity, site = PROXIMITY_MULTIPLIER, label
                break

    sev_c = severity ** 2
    head_c = 1 + math.log(reporters)
    age_c = 1 + days / 7
    score = sev_c * head_c * age_c * proximity

    return {
        "category": category,
        "hazard_lane": category in HAZARD_CATEGORIES,
        "severity": severity,
        "reporters": reporters,
        "days_pending": days,
        "severity_component": round(sev_c, 2),
        "headcount_component": round(head_c, 3),
        "ageing_component": round(age_c, 3),
        "proximity_component": proximity,
        "proximity_site": site,
        "final_score": round(score, 2),
        "formula": "S^2 x (1 + ln N) x (1 + D/7) x P",
    }


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def main(path):
    with open(path) as f:
        complaints = json.load(f)

    vectors, backend, backend_label = build_embedder(
        [c["description_en"] for c in complaints])
    clusters, merge_log = deduplicate(complaints, vectors, backend)

    scored = []
    for cl in clusters:
        p = priority(cl)
        scored.append((p, cl))
    scored.sort(key=lambda x: (not x[0]["hazard_lane"], -x[0]["final_score"]))

    print("=" * 74)
    print("NagarAI  --  deduplication and priority run")
    print("=" * 74)
    print(f"Embedding backend : {backend_label}")
    print(f"Complaints in     : {len(complaints)}")
    print(f"Distinct issues   : {len(clusters)}")
    print(f"Duplicates merged : {len(complaints) - len(clusters)}")

    print("\n" + "-" * 74)
    print("MERGE DECISIONS (every merge is explainable)")
    print("-" * 74)
    for m in merge_log:
        print(f"  {m['merged'][0]} + {m['merged'][1]}  [{m['category']}]  "
              f"{m['rule']}")
        print(f"      {m['reason']}")

    print("\n" + "-" * 74)
    print("PRIORITISED QUEUE")
    print("-" * 74)

    rank = 0
    for lane_flag, lane_title in [
        (True, "HAZARD LANE  --  worked first regardless of report count"),
        (False, "ROUTINE LANE  --  ranked by priority score"),
    ]:
        lane_items = [(p, cl) for p, cl in scored if p["hazard_lane"] == lane_flag]
        if not lane_items:
            continue
        print(f"\n{lane_title}")
        for p, cl in lane_items:
            rank += 1
            ids = ", ".join(c["id"] for c in cl)
            print(f"\n#{rank}  score {p['final_score']}   {p['category']}")
            print(f"     {cl[0]['description_en']}")
            print(f"     complaints: {ids}")
            print(f"     S={p['severity']} (S^2={p['severity_component']})  "
                  f"N={p['reporters']} (1+lnN={p['headcount_component']})  "
                  f"D={p['days_pending']} (1+D/7={p['ageing_component']})  "
                  f"P={p['proximity_component']}")
            if p["proximity_site"]:
                print(f"     proximity uplift: within 200 m of {p['proximity_site']}")

    print("\n" + "=" * 74)
    print("WORKED EXAMPLE FROM THE PROBLEM STATEMENT")
    print("=" * 74)
    pothole = 2 ** 2 * (1 + math.log(40)) * (1 + 6 / 7)
    wire = 5 ** 2 * (1 + math.log(2)) * (1 + 1 / 7)
    print(f"  40-report pothole  S=2 N=40 D=6  ->  {pothole:.2f}")
    print(f"   2-report live wire S=5 N=2  D=1  ->  {wire:.2f}  (+ hazard lane)")
    print("  Severity squared and headcount damped: the hazard wins on score")
    print("  alone, and the hazard lane makes it structural rather than lucky.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/seed_complaints.json")