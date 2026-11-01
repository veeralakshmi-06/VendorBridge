
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Project root
project_root = Path(__file__).resolve().parent.parent

# Files
faiss_file = project_root / "vector_store" / "faiss_index.bin"
metadata_file = project_root / "vector_store" / "chunks_metadata.json"

print("=" * 60)
print("PHASE 5 - SEMANTIC RETRIEVAL")
print("=" * 60)

# Load FAISS index
print("\nLoading FAISS index...")
index = faiss.read_index(str(faiss_file))
print(f"Number of vectors: {index.ntotal}")

# Load metadata
with open(metadata_file, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print(f"Number of metadata records: {len(metadata)}")

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded successfully.")

# User query
query = input("\nEnter your question: ").strip()
query_lower = query.lower()

# Choose preferred source based on question
procurement_keywords = [
    "procurement", "contract award", "award amount",
    "supplier name", "contract value", "award date",
    "contract description", "which supplier", "how much"
]

if any(word in query_lower for word in procurement_keywords):
    preferred_type = "csv"
else:
    preferred_type = "pdf"

print(f"\nPreferred source type: {preferred_type.upper()}")

# Convert query to embedding
query_embedding = model.encode(
    [query],
    convert_to_numpy=True
).astype("float32")

# Retrieve candidates from FAISS
k = min(50, index.ntotal)
distances, indices = index.search(query_embedding, k)

# Keep preferred sources first, followed by other sources
results = []

for distance, idx in zip(distances[0], indices[0]):
    if idx < 0 or idx >= len(metadata):
        continue

    chunk = metadata[idx]
    results.append((distance, chunk))

preferred_results = [
    item for item in results
    if item[1].get("source_type") == preferred_type
]

other_results = [
    item for item in results
    if item[1].get("source_type") != preferred_type
]

final_results = (preferred_results + other_results)[:5]

# Display results
print("\n" + "=" * 60)
print("TOP 5 RETRIEVED RESULTS")
print("=" * 60)

for rank, (distance, chunk) in enumerate(final_results, start=1):
    print(f"\n--- Result {rank} ---")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Source: {chunk['source']}")
    print(f"Source Type: {chunk['source_type']}")
    print(f"Distance: {distance:.4f}")
    print("\nContent:")
    print(chunk["text"][:1000])

print("\n" + "=" * 60)
print("PHASE 5 RETRIEVAL TEST COMPLETED")
print("=" * 60)

