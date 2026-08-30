"""
migrate_to_pinecone.py

One-time migration script for v2.0 deployment. Reads every chunk
already stored in the local Chroma vector store, including its
already-computed embedding, and uploads it to Pinecone.

This does NOT re-embed anything. The whole point is to reuse the
exact embeddings already produced by the local BGE model, so the
diagnosed and fixed retrieval accuracy carries over unchanged, only
where the vectors are stored changes.

Run once, from the project root:
    python pipeline/migrate_to_pinecone.py
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

from dotenv import load_dotenv
from pinecone import Pinecone

from retrieve_answer import load_vector_store
from load_documents import COMPANIES

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-report-assistant")
BATCH_SIZE = 100


def migrate():
    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY not found. Check that .env exists in the "
            "project root and contains PINECONE_API_KEY=your_key."
        )

    print("Loading local Chroma vector store...")
    vector_store = load_vector_store()

    print(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    total_uploaded = 0

    for company_name in COMPANIES.keys():
        print(f"\nReading chunks for {company_name} from Chroma...")

        raw = vector_store._collection.get(
            where={"company": company_name},
            include=["documents", "metadatas", "embeddings"],
        )

        ids = raw["ids"]
        documents = raw["documents"] or []
        metadatas = raw["metadatas"] or []
        embeddings = raw["embeddings"]

        count = len(ids)
        print(f"  Found {count} chunks. Uploading to Pinecone in batches of {BATCH_SIZE}...")

        for start in range(0, count, BATCH_SIZE):
            end = min(start + BATCH_SIZE, count)
            batch_vectors = []

            for i in range(start, end):
                # Pinecone doesn't store document text automatically like
                # Chroma does, so the chunk's own text is saved inside its
                # metadata, alongside company, sector, and page, so it can
                # be retrieved back after a search.
                metadata = dict(metadatas[i])
                metadata["text"] = documents[i]

                batch_vectors.append({
                    "id": ids[i],
                    "values": [float(x) for x in embeddings[i]],
                    "metadata": metadata,
                })

            index.upsert(vectors=batch_vectors, namespace="")
            print(f"    Uploaded {end}/{count}")

        total_uploaded += count

    print(f"\nDone. Total chunks uploaded to Pinecone: {total_uploaded}")

    stats = index.describe_index_stats()
    print(f"Pinecone index now reports: {stats['total_vector_count']} total vectors")


if __name__ == "__main__":
    migrate()
