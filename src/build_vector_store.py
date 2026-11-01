import json
import numpy as np
import faiss
from pathlib import Path

# Project root
project_root = Path(__file__).resolve().parent.parent

# Input files
embedding_file = project_root / "vector_store" / "embeddings.npy"
metadata_file = project_root / "vector_store" / "chunks_metadata.json"

# Output file
faiss_file = project_root / "vector_store" / "faiss_index.bin"

print("=" * 60)
print("PHASE 4 - FAISS VECTOR DATABASE")
print("=" * 60)

# Load embeddings
print("\nLoading embeddings...")

embeddings = np.load(embedding_file)

print(f"Number of embeddings: {embeddings.shape[0]}")
print(f"Embedding dimensions: {embeddings.shape[1]}")

# Convert to float32
embeddings = embeddings.astype("float32")

# Create FAISS index
print("\nCreating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add embeddings
index.add(embeddings)

print("FAISS index created successfully.")
print(f"Number of vectors in index: {index.ntotal}")

# Load metadata
print("\nLoading chunk metadata...")

with open(metadata_file, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print(f"Number of metadata records: {len(metadata)}")

# Save FAISS index
faiss.write_index(index, str(faiss_file))

print("\nFAISS index saved to:")
print(faiss_file)

print("\n" + "=" * 60)
print("PHASE 4 COMPLETED")
print("=" * 60)