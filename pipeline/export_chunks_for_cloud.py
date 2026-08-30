"""
export_chunks_for_cloud.py

Exports every chunk's text and metadata (company, sector, page) from
the local Chroma vector store into lightweight JSON files, one per
company. Deliberately excludes the embedding vectors themselves,
since BM25 keyword search only needs plain text, not the heavy
numeric vectors.

Why this exists: the deployed app uses Pinecone for vector search,
but Pinecone isn't built for "give me all of company X's chunks"
the way Chroma is. Rather than re-fetch thousands of chunks from
Pinecone on every app startup just to build a keyword index, this
data ships as part of the app's own code instead, small, fast, and
available instantly with no network call needed for the BM25 half
of hybrid search.

Run once, from the project root:
    python pipeline/export_chunks_for_cloud.py
"""

import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__)))

from retrieve_answer import load_vector_store
from load_documents import COMPANIES

OUTPUT_DIR = "data/chunks_export"


def export_chunks():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading local Chroma vector store...")
    vector_store = load_vector_store()

    total_exported = 0

    for company_name in COMPANIES.keys():
        print(f"\nExporting chunks for {company_name}...")

        raw = vector_store._collection.get(
            where={"company": company_name},
            include=["documents", "metadatas"],
        )

        ids = raw["ids"]
        documents = raw["documents"] or []
        metadatas = raw["metadatas"] or []

        chunks = []
        for i in range(len(ids)):
            chunks.append({
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
            })

        safe_name = company_name.replace(" ", "_")
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Exported {len(chunks)} chunks to {output_path} ({size_kb:.0f} KB)")
        total_exported += len(chunks)

    print(f"\nDone. Total chunks exported: {total_exported}")


if __name__ == "__main__":
    export_chunks()
