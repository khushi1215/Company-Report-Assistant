"""
retrieve_answer.py

Takes a company name and a question, searches only that company's
chunks in Chroma, and asks a local LLM (Llama 3.1, via Ollama) to
answer using only the retrieved chunks.

The prompt explicitly tells the model not to use outside knowledge,
and to say so if the answer isn't in the retrieved chunks. This is
the core of what makes this a grounded, RAG based answer instead of
a normal chatbot answer.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_NAME = "llama3.2"
NUM_CHUNKS_TO_RETRIEVE = 3

PROMPT_TEMPLATE = """You are answering questions using only the context provided below, taken from {company}'s annual report. Do not use any outside knowledge. If the answer is not clearly in the context, say you could not find that information in the report, do not guess.

Write your answer in clear, complete sentences of your own. The context below may be messy, extracted from tables, stat boxes, or multi-column page layouts, so it may not read like normal prose. Do not copy fragments or phrases directly from the context as-is. Read it, understand what it says, and explain it plainly in your own words instead.

Context from the report:
{context}

Question: {question}

Answer:"""


def load_vector_store():
    """
    Loads the existing Chroma vector store from disk.
    This assumes chunk_and_embed.py has already been run once.

    Uses HuggingFaceBgeEmbeddings specifically, not the generic
    HuggingFaceEmbeddings class, because BGE models are trained to
    expect a short instruction prefix added to queries (but not to
    the stored document chunks themselves) for best retrieval
    accuracy. Skipping this quietly loses part of the accuracy this
    model switch was meant to gain.
    """
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        query_instruction="Represent this sentence for searching relevant passages: ",
    )
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings,
    )
    return vector_store


def retrieve_chunks(vector_store, company, question):
    """
    Searches the vector store for chunks relevant to the question,
    filtered to only the selected company using Chroma's metadata
    filter. This is what keeps answers from mixing companies together.
    """
    results = vector_store.similarity_search(
        query=question,
        k=NUM_CHUNKS_TO_RETRIEVE,
        filter={"company": company},
    )
    return results


def get_company_documents(vector_store, company):
    """
    Pulls every stored chunk for one company directly out of Chroma,
    reconstructed as LangChain Document objects. BM25 keyword search
    needs the actual chunk texts in memory, not just the vector index,
    so this rebuilds that list from what's already stored, no need to
    re-run the chunking pipeline.
    """
    raw = vector_store._collection.get(
        where={"company": company},
        include=["documents", "metadatas"],
    )

    documents = []
    for text, metadata in zip(raw["documents"] or [], raw["metadatas"] or []):
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def bm25_preprocess(text):
    """
    Custom tokenizer for BM25 keyword search. Lowercases the text and
    expands the '%' symbol into the word 'percent', so a chunk that
    literally says "29%" still matches a question asking about
    "percentage", which BM25's plain word-matching would otherwise
    miss (confirmed as a real cause of missed retrieval, see
    KNOWLEDGE.md). Applied consistently to both the stored chunks
    and the incoming question, so the comparison stays fair.
    """
    text = text.lower().replace("%", " percent ")
    return text.split()


def build_hybrid_retriever(vector_store, company, k=NUM_CHUNKS_TO_RETRIEVE):
    """
    Combines two different search methods into one:
    - Vector search: finds chunks that mean something similar to the
      question, even with different wording.
    - BM25 keyword search: finds chunks that literally contain the
      question's words, which catches cases where a chunk's meaning
      got scrambled during PDF extraction (see KNOWLEDGE.md) and so
      ranks poorly on pure vector similarity, even though it still
      contains the exact right words.

    Each method searches slightly wider than the final answer needs
    (k + 2), then EnsembleRetriever merges and re-ranks both lists
    before we trim down to the final chunk count.
    """
    search_width = k + 2

    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": search_width, "filter": {"company": company}}
    )

    company_docs = get_company_documents(vector_store, company)
    bm25_retriever = BM25Retriever.from_documents(company_docs, preprocess_func=bm25_preprocess)
    bm25_retriever.k = search_width

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    return hybrid_retriever


def retrieve_chunks_hybrid(vector_store, company, question, hybrid_retriever=None, k=NUM_CHUNKS_TO_RETRIEVE):
    """
    Same purpose as retrieve_chunks, but uses hybrid (vector + keyword)
    search instead of vector search alone. A pre-built hybrid_retriever
    can be passed in to avoid rebuilding the BM25 index on every single
    question, since that rebuild is the slow part, not the search itself.
    """
    if hybrid_retriever is None:
        hybrid_retriever = build_hybrid_retriever(vector_store, company, k=k)

    results = hybrid_retriever.invoke(question)
    return results[:k]



def build_context_string(chunks):
    """
    Combines retrieved chunks into one text block for the prompt,
    with each chunk labeled by its page number so the model's
    answer can be checked against a specific page.
    """
    parts = []
    for chunk in chunks:
        page = chunk.metadata.get("page", "unknown")
        parts.append(f"[Page {page}]\n{chunk.page_content}")
    return "\n\n".join(parts)


def get_answer(company, question, vector_store, llm):
    """
    Full retrieval + answer flow for one question.
    Returns the answer text and the list of source chunks used,
    so the app can show both the answer and the citation detail.
    """
    chunks = retrieve_chunks(vector_store, company, question)

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


# Quick manual test, run this file directly to try one question
# before wiring it into the Streamlit app. Using a question that
# failed during evaluation (Sun Pharma US business revenue, real
# content confirmed present on page 36 but missed by pure vector
# search) to check whether hybrid search actually helps.
if __name__ == "__main__":
    print("Loading vector store...")
    vector_store = load_vector_store()

    print(f"Loading LLM: {LLM_MODEL_NAME} (via Ollama)")
    llm = ChatOllama(model=LLM_MODEL_NAME)

    test_company = "Sun Pharma"
    test_question = "What percentage of Sun Pharma's revenue comes from its US business?"

    print(f"\nCompany: {test_company}")
    print(f"Question: {test_question}")
    print("\nBuilding hybrid retriever (vector + keyword search)...")

    hybrid_retriever = build_hybrid_retriever(vector_store, test_company)
    chunks = retrieve_chunks_hybrid(vector_store, test_company, test_question, hybrid_retriever=hybrid_retriever)

    print(f"\nRetrieved {len(chunks)} chunks, from pages:")
    for c in chunks:
        print(f"  Page {c.metadata.get('page')}")

    context = build_context_string(chunks)
    prompt = PROMPT_TEMPLATE.format(company=test_company, context=context, question=test_question)

    print("\nGenerating answer, this may take a moment on a local model...\n")
    response = llm.invoke(prompt)

    print("Answer:")
    print(response.content)
