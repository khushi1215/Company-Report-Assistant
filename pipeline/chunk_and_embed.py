"""
chunk_and_embed.py

Takes the page-by-page text from load_documents.py, splits it into
smaller chunks, converts each chunk into an embedding using a free
Hugging Face model, and stores everything in Chroma.

Each chunk keeps 3 pieces of metadata attached:
- company: which company this chunk came from
- sector: that company's sector
- page: which page in the original PDF this chunk came from

This metadata is what makes the dropdown filtering and page
citations possible later.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from load_documents import load_all_companies

CHUNK_SIZE = 1300
CHUNK_OVERLAP = 200
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def build_documents(all_company_data):
    """
    Converts the page-by-page text into LangChain Document objects,
    split into chunks, with metadata attached to each chunk.

    Splitting happens per page, not on the whole report at once.
    This keeps the page number accurate for every chunk, since a
    chunk never crosses a page boundary.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_documents = []

    for company_name, info in all_company_data.items():
        sector = info["sector"]
        for page_data in info["pages"]:
            page_number = page_data["page"]
            page_text = page_data["text"]

            chunks = splitter.split_text(page_text)

            for chunk_text in chunks:
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "company": company_name,
                        "sector": sector,
                        "page": page_number,
                    },
                )
                all_documents.append(doc)

        print(f"  {company_name}: chunked into pieces, running total {len(all_documents)} chunks")

    return all_documents


def embed_and_store(documents, batch_size=200):
    """
    Embeds all chunks and stores them in a Chroma vector store on disk
    at VECTOR_STORE_PATH. This only needs to run once, unless the
    source PDFs or chunk settings change.

    Processes documents in batches instead of one single call, so
    real progress prints as it goes. A bigger, more accurate
    embedding model (BAAI/bge-base-en-v1.5) genuinely takes a long
    time on CPU-only hardware, and running silently for 30+ minutes
    with no feedback was a real usability gap worth fixing, not
    just a display nicety.
    """
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    print("(First run downloads the model, may take a minute.)")

    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    total = len(documents)
    print(f"\nEmbedding {total} chunks and storing in Chroma...")
    print("(This can genuinely take 30-60+ minutes on CPU-only hardware")
    print("with this model size. Progress below updates as it runs.)\n")

    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings,
    )

    for start in range(0, total, batch_size):
        batch = documents[start:start + batch_size]
        vector_store.add_documents(batch)

        done = min(start + batch_size, total)
        pct = done / total * 100
        print(f"  Embedded {done}/{total} chunks ({pct:.1f}%)")

    print(f"\nDone. Vector store saved to '{VECTOR_STORE_PATH}'.")
    return vector_store


if __name__ == "__main__":
    print("Step 1: Loading PDF text...")
    all_data = load_all_companies()

    print("\nStep 2: Splitting into chunks...")
    documents = build_documents(all_data)
    print(f"\nTotal chunks created across all companies: {len(documents)}")

    print("\nStep 3: Embedding and storing in Chroma...")
    embed_and_store(documents)
