"""
diagnose_chunk.py

One-off diagnostic script, not part of the main pipeline.
Prints every stored chunk for Sun Pharma that came from page 36,
so we can see exactly how the real answer text got split up.

Run from the project root:
    python diagnose_chunk.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from retrieve_answer import load_vector_store

vector_store = load_vector_store()

raw = vector_store._collection.get(
    where={"company": "Sun Pharma"},
    include=["documents", "metadatas"],
)

documents = raw["documents"] or []
metadatas = raw["metadatas"] or []

print(f"Total Sun Pharma chunks in store: {len(documents)}\n")

page_36_chunks = []
for text, metadata in zip(documents, metadatas):
    if metadata.get("page") == 36:
        page_36_chunks.append(text)

print(f"Chunks found from page 36: {len(page_36_chunks)}\n")

for i, chunk in enumerate(page_36_chunks, 1):
    print(f"--- Chunk {i} (page 36) ---")
    print(chunk)
    print()
