"""
retrieve_answer_cloud.py

Cloud version of retrieve_answer.py, used only when the app is
deployed. Same hybrid search idea as the local version (vector +
BM25 keyword search combined), just sourced differently:

- Vector search: Pinecone, instead of local Chroma
- Query embedding: Hugging Face's hosted Inference API, instead of
  a locally-loaded BGE model
- BM25 keyword search: built from the exported JSON chunk files
  (data/chunks_export/), instead of pulling chunks out of Chroma

Local development is unaffected by this file. It uses
retrieve_answer.py and Chroma exactly as before.
"""

import os
import json

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from pydantic import SecretStr

from prompt_template import PROMPT_TEMPLATE

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-report-assistant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "openai/gpt-oss-20b"
CHUNKS_EXPORT_DIR = "data/chunks_export"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
NUM_CHUNKS_TO_RETRIEVE = 3


class HFInferenceAPIEmbeddings(Embeddings):
    """
    A LangChain-compatible embeddings class that calls Hugging Face's
    hosted Inference API instead of loading a model locally. This is
    what keeps the deployed app from needing PyTorch or the BGE model
    weights in its own memory at all.

    Matches the same query-vs-document asymmetry used locally: only
    the question gets the BGE query instruction prefix, stored
    document text does not, since documents were already embedded
    that way during the original local build.
    """

    def __init__(self):
        self.client = InferenceClient(token=HF_API_TOKEN)

    def embed_query(self, text):
        result = self.client.feature_extraction(
            QUERY_INSTRUCTION + text,
            model=EMBEDDING_MODEL_NAME,
        )
        return [float(x) for x in result]

    def embed_documents(self, texts):
        # Not used for querying, only required to satisfy LangChain's
        # Embeddings interface. Document embeddings were already
        # computed locally and migrated to Pinecone directly.
        return [self.embed_query(t) for t in texts]


def load_cloud_vector_store():
    """
    Connects to the Pinecone index using the HF-hosted embeddings
    class above, so queries get embedded via the API automatically
    whenever a search is run through this vector store.
    """
    if not PINECONE_API_KEY or not HF_API_TOKEN:
        raise ValueError(
            "PINECONE_API_KEY and HF_API_TOKEN must both be set "
            "(as environment variables or Streamlit secrets) to use "
            "the cloud retrieval path."
        )

    embeddings = HFInferenceAPIEmbeddings()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    return vector_store


def load_company_documents_from_export(company):
    """
    Loads one company's chunks from the exported JSON file, for
    building the BM25 keyword retriever. No network call, no vector
    computation, just reading a small local file.
    """
    safe_name = company.replace(" ", "_")
    path = os.path.join(CHUNKS_EXPORT_DIR, f"{safe_name}.json")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents = []
    for chunk in chunks:
        documents.append(Document(page_content=chunk["text"], metadata=chunk["metadata"]))

    return documents


def clean_page_number(page):
    """
    Pinecone can return numeric metadata as a float (e.g. 37.0
    instead of 37), even though the same value comes back as a
    clean int from the local BM25/JSON path. Normalizes either case
    to a plain int for consistent, clean display.
    """
    try:
        return int(float(page))
    except (TypeError, ValueError):
        return page


def bm25_preprocess(text):
    """Same normalization used locally: expands '%' to 'percent' so
    keyword search matches questions phrased with the word instead
    of the symbol."""
    text = text.lower().replace("%", " percent ")
    return text.split()


def build_cloud_hybrid_retriever(vector_store, company, k=NUM_CHUNKS_TO_RETRIEVE):
    """
    Same hybrid search idea as the local version: vector search
    (Pinecone) combined with keyword search (BM25, built from the
    exported JSON), merged via EnsembleRetriever.
    """
    search_width = k + 2

    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": search_width, "filter": {"company": company}}
    )

    company_docs = load_company_documents_from_export(company)
    bm25_retriever = BM25Retriever.from_documents(company_docs, preprocess_func=bm25_preprocess)
    bm25_retriever.k = search_width

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    return hybrid_retriever


def retrieve_chunks_cloud(vector_store, company, question, hybrid_retriever=None, k=NUM_CHUNKS_TO_RETRIEVE):
    if hybrid_retriever is None:
        hybrid_retriever = build_cloud_hybrid_retriever(vector_store, company, k=k)

    results = hybrid_retriever.invoke(question)
    return results[:k]


def build_context_string(chunks):
    """
    Same purpose as the local version: combines retrieved chunks
    into one text block for the prompt, each labeled by page number.
    """
    parts = []
    for chunk in chunks:
        page = clean_page_number(chunk.metadata.get("page", "unknown"))
        parts.append(f"[Page {page}]\n{chunk.page_content}")
    return "\n\n".join(parts)


def get_cloud_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Check your .env file or Streamlit secrets.")
    return ChatGroq(model=GROQ_MODEL_NAME, api_key=SecretStr(GROQ_API_KEY))


def get_answer_cloud(company, question, vector_store, llm, hybrid_retriever=None):
    """
    Full cloud retrieval + answer flow for one question, mirroring
    get_answer in retrieve_answer.py exactly, just using Pinecone
    and Groq instead of Chroma and Ollama.
    """
    chunks = retrieve_chunks_cloud(vector_store, company, question, hybrid_retriever=hybrid_retriever)

    if not chunks:
        return {
            "answer": "No relevant content was found in this company's report for that question.",
            "sources": [],
        }

    context = build_context_string(chunks)
    prompt = PROMPT_TEMPLATE.format(company=company, context=context, question=question)

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": chunks,
    }


if __name__ == "__main__":
    print("Testing full cloud pipeline: Pinecone retrieval + Groq answer...\n")
    vector_store = load_cloud_vector_store()
    llm = get_cloud_llm()

    test_company = "Sun Pharma"
    test_question = "What percentage of Sun Pharma's revenue comes from its US business?"

    print(f"Company: {test_company}")
    print(f"Question: {test_question}\n")

    result = get_answer_cloud(test_company, test_question, vector_store, llm)

    print("Answer:")
    print(result["answer"])

    print("\nSources used:")
    for source in result["sources"]:
        page = clean_page_number(source.metadata.get("page"))
        print(f"  Page {page}")
