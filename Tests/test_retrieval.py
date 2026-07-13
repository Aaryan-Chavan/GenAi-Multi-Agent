import sys
from pathlib import Path

# ==========================================================
# PROJECT ROOT SETUP
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from Config.settings import (
    DUCKDB_FILE,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL
)

# ==========================================================
# CONFIG
# ==========================================================

DUCKDB_PATH = str(DUCKDB_FILE)
COLLECTION_NAME = QDRANT_COLLECTION
TOP_K = 20

# ==========================================================
# CONNECT QDRANT
# ==========================================================

print("\n" + "=" * 80)
print("CONNECTING TO QDRANT")
print("=" * 80)

client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)

print("Connected Successfully")

print("\n" + "=" * 80)
print("QDRANT PAYLOAD DEBUG SAMPLE")
print("=" * 80)

sample = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=5,
    with_payload=True,
    with_vectors=False
)

points = sample[0]  # first element = points list

for i, point in enumerate(points, start=1):
    print(f"\nPOINT {i}")
    print("-" * 50)

    payload = point.payload or {}

    for k, v in payload.items():
        print(f"{k}: {v}")

print("\nDEBUG COMPLETE")
print("=" * 80)

# ==========================================================
# LOAD EMBEDDING MODEL
# ==========================================================

print("\nLoading Embedding Model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Model Loaded.")

# ==========================================================
# QUERY INTENT DETECTOR (OPTION 2 CORE LOGIC)
# ==========================================================

def build_filter(query: str):
    """
    Convert natural language query → structured Qdrant filter
    """

    q = query.lower()

    must_conditions = []

    # SENTIMENT FILTERS
    if "negative" in q:
        must_conditions.append(
            {"key": "sentiment_label", "match": {"value": "negative"}}
        )

    if "positive" in q:
        must_conditions.append(
            {"key": "sentiment_label", "match": {"value": "positive"}}
        )

    if "neutral" in q:
        must_conditions.append(
            {"key": "sentiment_label", "match": {"value": "neutral"}}
        )

    # COMPLAINT FILTER
    if "complaint" in q or "issues" in q:
        must_conditions.append(
            {"key": "is_complaint", "match": {"value": True}}
        )

    # URGENCY FILTER
    if "urgent" in q:
        must_conditions.append(
            {"key": "is_urgent", "match": {"value": True}}
        )

    # SEVERITY FILTER
    if "critical" in q:
        must_conditions.append(
            {"key": "severity", "match": {"value": "CRITICAL"}}
        )

    if not must_conditions:
        return None

    return {"must": must_conditions}

# ==========================================================
# SEARCH LOOP
# ==========================================================

print("\n" + "=" * 80)
print("READY FOR SEARCH (HYBRID MODE)")
print("Type 'exit' to quit")
print("=" * 80)

while True:

    query = input("\nEnter search query: ").strip()

    if query.lower() == "exit":
        break

    if not query:
        continue

    # ======================================================
    # EMBEDDING
    # ======================================================

    query_vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # ======================================================
    # BUILD FILTER (OPTION 2)
    # ======================================================

    query_filter = build_filter(query)

    # ======================================================
    # QDRANT SEARCH (HYBRID)
    # ======================================================

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter   # <-- OPTION 2 ENABLED
    )

    results = response.points

    # ======================================================
    # OUTPUT
    # ======================================================

    print("\n" + "=" * 80)
    print(f"SEARCH RESULTS (Top {TOP_K})")
    print("=" * 80)

    if not results:
        print("No results found.")
        continue

    for i, hit in enumerate(results, start=1):

        payload = hit.payload or {}

        print(f"\nResult {i}")
        print("-" * 60)

        print(f"Score: {hit.score:.4f}")

        # prioritized fields
        priority_keys = [
            "chunk_text",
            "sentiment_label",
            "sentiment_score",
            "severity",
            "intent",
            "complaint_type",
            "is_complaint",
            "is_urgent",
            "aspects_detected",
            "topics_keywords",
            "record_id",
            "chunk_id"
        ]

        printed = set()

        for key in priority_keys:
            if key in payload:
                print(f"{key}: {payload[key]}")
                printed.add(key)

        for key, value in payload.items():
            if key not in printed:
                print(f"{key}: {value}")

print("\nFinished.")