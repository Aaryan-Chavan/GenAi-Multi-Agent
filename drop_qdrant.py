from Storage.qdrant_loader import QdrantLoader
loader = QdrantLoader()
loader.delete_collection()
print("Done")