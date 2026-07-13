from qdrant_client import QdrantClient
from pprint import pprint

# --------------------------------------------------
# Connect to Qdrant
# --------------------------------------------------
client = QdrantClient(
    host="localhost",
    port=6333
)

# --------------------------------------------------
# List Collections
# --------------------------------------------------
print("=" * 80)
print("AVAILABLE COLLECTIONS")
print("=" * 80)

collections = client.get_collections()

for collection in collections.collections:
    print(collection.name)

# --------------------------------------------------
# Change this
# --------------------------------------------------
COLLECTION_NAME = "documents"

print("\n" + "=" * 80)
print("COLLECTION INFO")
print("=" * 80)

info = client.get_collection(COLLECTION_NAME)

pprint(info)

# --------------------------------------------------
# Number of vectors
# --------------------------------------------------
count = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
)

print("\nTotal Points:", count.count)

# --------------------------------------------------
# Payload schema
# --------------------------------------------------
print("\n" + "=" * 80)
print("PAYLOAD SCHEMA")
print("=" * 80)

try:
    pprint(info.payload_schema)
except Exception:
    print("No payload schema found.")

# --------------------------------------------------
# Vector configuration
# --------------------------------------------------
print("\n" + "=" * 80)
print("VECTOR CONFIG")
print("=" * 80)

try:
    pprint(info.config.params.vectors)
except Exception:
    pass

# --------------------------------------------------
# Fetch sample records
# --------------------------------------------------
print("\n" + "=" * 80)
print("SAMPLE STORED POINTS")
print("=" * 80)

points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=5,
    with_payload=True,
    with_vectors=False
)

for i, point in enumerate(points, 1):

    print(f"\nPoint {i}")
    print("-" * 60)

    print("ID:")
    pprint(point.id)

    print("\nPayload:")
    pprint(point.payload)

# --------------------------------------------------
# Fetch vectors too (optional)
# --------------------------------------------------
print("\n" + "=" * 80)
print("POINTS WITH VECTORS")
print("=" * 80)

points, _ = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=2,
    with_payload=True,
    with_vectors=True
)

for point in points:

    print("\nID:", point.id)

    if isinstance(point.vector, dict):
        for name, vec in point.vector.items():
            print(f"Vector '{name}' length =", len(vec))
    else:
        print("Vector length =", len(point.vector))

    print("Payload:")
    pprint(point.payload)