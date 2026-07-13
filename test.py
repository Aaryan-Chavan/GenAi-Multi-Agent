from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "documents"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5
SAMPLE_SIZE = 100

# ============================================================
# CONNECT
# ============================================================

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer(MODEL_NAME)

# ============================================================
# LOAD SAMPLE POINTS
# ============================================================

points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=SAMPLE_SIZE,
    with_payload=True,
    with_vectors=True,
)

print(f"\nLoaded {len(points)} points")

if not points:
    raise Exception("Collection is empty.")

# ============================================================
# TEST 1 : VECTOR DIMENSION
# ============================================================

expected_dim = model.get_embedding_dimension()

print("\n" + "=" * 70)
print("TEST 1 : VECTOR DIMENSION")
print("=" * 70)

bad = False

for p in points:
    if len(p.vector) != expected_dim:
        bad = True
        print(f"Wrong dimension for {p.id}: {len(p.vector)}")

if not bad:
    print(f"PASS ✓ All vectors are {expected_dim}-dimensional")

# ============================================================
# TEST 2 : VECTOR NORMS
# ============================================================

print("\n" + "=" * 70)
print("TEST 2 : VECTOR NORMS")
print("=" * 70)

vectors = np.array([p.vector for p in points])

norms = np.linalg.norm(vectors, axis=1)

print(f"Average norm : {norms.mean():.6f}")
print(f"Minimum norm : {norms.min():.6f}")
print(f"Maximum norm : {norms.max():.6f}")

# BGE embeddings are normalized, so this is expected.
if norms.std() < 1e-5:
    print("✓ Embeddings are L2-normalized (expected for BGE models).")

# ============================================================
# TEST 3 : DUPLICATE VECTORS
# ============================================================

print("\n" + "=" * 70)
print("TEST 3 : DUPLICATE VECTORS")
print("=" * 70)

duplicates = 0

for i in range(len(vectors)):
    for j in range(i + 1, len(vectors)):
        sim = np.dot(vectors[i], vectors[j])

        if sim > 0.99999:
            duplicates += 1

print("Near-identical vector pairs:", duplicates)

# ============================================================
# TEST 4 : SELF RETRIEVAL
# ============================================================

print("\n" + "=" * 70)
print("TEST 4 : SELF RETRIEVAL")
print("=" * 70)

correct = 0

for point in points[:20]:

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=point.vector,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )

    hits = response.points

    if hits and hits[0].id == point.id:
        correct += 1

print(f"Self Retrieval Accuracy: {correct}/20 ({correct*5}%)")

# ============================================================
# TEST 5 : TEXT RETRIEVAL
# ============================================================

print("\n" + "=" * 70)
print("TEST 5 : TEXT RETRIEVAL")
print("=" * 70)

payload = points[0].payload

text = None

for key in [
    "text",
    "chunk_text",
    "content",
    "page_content",
    "body",
    "document",
]:
    if key in payload:
        text = payload[key]
        break

if text is None:
    print("Could not find a text field in payload.")
else:

    embedding = model.encode(text, normalize_embeddings=True).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=TOP_K,
    )

    hits = response.points

    print("\nQuery:\n")
    print(text[:300])

    print("\nTop Results:\n")

    for rank, hit in enumerate(hits, start=1):

        retrieved = ""

        for key in [
            "text",
            "chunk_text",
            "content",
            "page_content",
            "body",
            "document",
        ]:
            if key in hit.payload:
                retrieved = hit.payload[key]
                break

        print("-" * 60)
        print(f"Rank : {rank}")
        print(f"Score: {hit.score:.4f}")
        print(retrieved[:200])

# ============================================================
# TEST 6 : SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TEST 6 : SCORE DISTRIBUTION")
print("=" * 70)

scores = []

for point in points[:20]:

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=point.vector,
        limit=TOP_K,
    )

    scores.extend([x.score for x in response.points])

scores = np.array(scores)

print(f"Average similarity : {scores.mean():.4f}")
print(f"Minimum similarity : {scores.min():.4f}")
print(f"Maximum similarity : {scores.max():.4f}")

print("\n" + "=" * 70)
print("OVERALL HEALTH CHECK")
print("=" * 70)

print(f"Embedding Dimension : {expected_dim}")
print(f"Vector Count Tested : {len(points)}")
print(f"Duplicate Pairs      : {duplicates}")
print(f"Self Retrieval       : {correct}/20")
print(f"Average Similarity   : {scores.mean():.4f}")

if correct == 20:
    print("\n✅ Embeddings and Qdrant index appear healthy.")
elif correct >= 18:
    print("\n🟢 Embeddings appear good with minor retrieval variations.")
else:
    print("\n🔴 Retrieval quality needs investigation.")