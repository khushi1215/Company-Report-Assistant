"""
check_chunk_count.py

One-off diagnostic script, not part of the main pipeline.
Prints the total number of chunks currently stored in the vector
store, and a breakdown per company. Used to get the real chunk
count after the chunk size was changed from 1000/150 to 1300/200,
since that number was never captured at the time.

Run from the project root:
    python diagnostics/check_chunk_count.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from retrieve_answer import load_vector_store
from load_documents import COMPANIES

vector_store = load_vector_store()

print("Chunk counts per company:\n")

total = 0
for company_name in COMPANIES.keys():
    raw = vector_store._collection.get(
        where={"company": company_name},
        include=[],
    )
    count = len(raw["ids"])
    total += count
    print(f"  {company_name}: {count} chunks")

print(f"\nTotal chunks across all companies: {total}")
