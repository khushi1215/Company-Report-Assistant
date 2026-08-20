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
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2"
NUM_CHUNKS_TO_RETRIEVE = 3

PROMPT_TEMPLATE = """You are answering questions using only the context provided below, taken from {company}'s annual report. Do not use any outside knowledge. If the answer is not clearly in the context, say you could not find that information in the report, do not guess.

Context from the report:
{context}

Question: {question}

Answer:"""


def load_vector_store():
    """
    Loads the existing Chroma vector store from disk.
    This assumes chunk_and_embed.py has already been run once.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
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
# before wiring it into the Streamlit app.
if __name__ == "__main__":
    print("Loading vector store...")
    vector_store = load_vector_store()

    print(f"Loading LLM: {LLM_MODEL_NAME} (via Ollama)")
    llm = ChatOllama(model=LLM_MODEL_NAME)

    test_company = "HDFC Bank"
    test_question = "What did the company say about digital banking initiatives?"

    print(f"\nCompany: {test_company}")
    print(f"Question: {test_question}")
    print("\nGenerating answer, this may take a moment on a local model...\n")

    result = get_answer(test_company, test_question, vector_store, llm)

    print("Answer:")
    print(result["answer"])

    print("\nSources used:")
    for source in result["sources"]:
        print(f"  Page {source.metadata.get('page')}")