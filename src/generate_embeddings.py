import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Project root
project_root = Path(__file__).resolve().parent.parent

# Input file
chunks_file = project_root / "chunks" / "all_chunks.json"

# Output folder
vector_store = project_root / "vector_store"
vector_store.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 3 - EMBEDDING GENERATION")
print("=" * 60)

# Load chunks
with open(chunks_file, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"\nNumber of chunks: {len(chunks)}")

# Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")

# Extract text
texts = [chunk["text"] for chunk in chunks]

# Generate embeddings
print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nEmbedding generation completed.")

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")

# Save embeddings
embedding_file = vector_store / "embeddings.npy"

np.save(embedding_file, embeddings)

# Save chunk metadata separately
metadata_file = vector_store / "chunks_metadata.json"

with open(metadata_file, "w", encoding="utf-8") as file:
    json.dump(chunks, file, indent=2, ensure_ascii=False)

print("\nEmbeddings saved to:")
print(embedding_file)

print("\nChunk metadata saved to:")
print(metadata_file)

print("\n" + "=" * 60)
print("PHASE 3 COMPLETED")
print("=" * 60)