"""
diagnose_ranking.py

One-off diagnostic script, not part of the main pipeline.
Shows exactly where the page-36 chunk (the one with the real answer)
ranks in vector search vs keyword search, separately, for the
Sun Pharma US business question. This tells us which search method
is missing it, and by how much, instead of guessing.

Run from the project root:
    python diagnose_ranking.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from retrieve_answer import load_vector_store, get_company_documents
from langchain_community.retrievers import BM25Retriever

vector_store = load_vector_store()
company = "Sun Pharma"
question = "What percentage of Sun Pharma's revenue comes from its US business?"

print(f"Question: {question}\n")

# --- Vector search, wide net, top 15 ---
print("=== VECTOR SEARCH: top 15 results ===")
vector_results = vector_store.similarity_search(
    query=question, k=15, filter={"company": company}
)
for i, doc in enumerate(vector_results, 1):
    page = doc.metadata.get("page")
    preview = doc.page_content[:60].replace("\n", " ")
    marker = "  <-- TARGET PAGE" if page == 36 else ""
    print(f"{i}. Page {page}: {preview}...{marker}")

# --- BM25 keyword search, wide net, top 15 ---
print("\n=== KEYWORD (BM25) SEARCH: top 15 results ===")
company_docs = get_company_documents(vector_store, company)
bm25 = BM25Retriever.from_documents(company_docs)
bm25.k = 15
bm25_results = bm25.invoke(question)
for i, doc in enumerate(bm25_results, 1):
    page = doc.metadata.get("page")
    preview = doc.page_content[:60].replace("\n", " ")
    marker = "  <-- TARGET PAGE" if page == 36 else ""
    print(f"{i}. Page {page}: {preview}...{marker}")

print("\nIf page 36 doesn't appear in either list above even at rank 15,")
print("that tells us this needs a different fix than widening the search.")
