# Company Report Assistant

![Status](https://img.shields.io/badge/status-live-brightgreen)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B)
![LangChain](https://img.shields.io/badge/langchain-hybrid%20RAG-1C3C3C)
![Deployment](https://img.shields.io/badge/deployed-Render%20%2B%20Docker-46E3B7)

Ask questions about a company's annual report in plain English, get answers pulled straight from the actual document, not a guess.

**🔗 Live demo:** [company-report-assistant.onrender.com](https://company-report-assistant.onrender.com)

Runs on a fully free, hosted architecture: [Pinecone](https://pinecone.io) for vector storage, Hugging Face's Inference API for query embedding, and [Groq](https://groq.com) for the LLM, deployed via Docker on Render. Local development uses a different, fully offline stack (Chroma, a locally-loaded embedding model, and Llama 3.2 via Ollama). Both are real, working, and documented, see [KNOWLEDGE.md](./KNOWLEDGE.md) for the full reasoning behind why the project has two architectures and how each was verified.

**Note:** this runs on Render's free tier, which spins down after periods of inactivity. The first request after a period of no traffic may take 30-60 seconds to wake back up, after that, answers generate in under 10 seconds.

---

## Highlights

- Answers are grounded in real annual report text, not general knowledge
- Every answer streams in live and comes with a source you can check, down to the page
- Covers 5 Indian companies across 5 different sectors: Banking, IT, FMCG, Energy, and Pharma
- Simple, no technical setup needed to use it, light and dark mode included
- Runs on a fully free architecture, hosted services for the live demo, local and offline for development, no paid API anywhere

---

## Demo

*(Screenshot or short GIF of the app in use goes here once built.)*

---

## Installation

Want to just try it? Use the live demo linked above, no setup needed.

To run it locally instead (local development uses a different, fully offline stack):

```bash
git clone https://github.com/khushi1215/company-report-assistant.git
cd company-report-assistant
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

This project runs on a free, local LLM (no API key or paid account needed). Install [Ollama](https://ollama.com), then pull the model:

```bash
ollama pull llama3.2
```

---

## Usage

```bash
streamlit run app.py
```

Then in the app:
1. Pick a company (shown as buttons across the top)
2. Type a question, for example: *"What did the company say about supply chain risk this year?"*
3. Watch the answer stream in, check the source line below it, or open the Sources tab for the full retrieved text and page number

---

## Tech Stack

This project runs two different, fully working architectures, one for local development, one for the live deployment. Both use the same RAG pipeline and LangChain orchestration underneath, only where the heavy pieces run differs. The full reasoning behind this, including real measured memory numbers, is documented in [KNOWLEDGE.md](./KNOWLEDGE.md).

**Local development:**
- **LangChain** — orchestrates the retrieval and answer pipeline
- **Chroma** — stores document chunks as searchable embeddings
- **Hugging Face sentence-transformers** — turns text into embeddings, running locally
- **Llama 3.2, via Ollama** — generates answers, running locally, free, no API key

**Live deployment ([Docker](./Dockerfile) on Render):**
- **Pinecone** — hosted vector database, stores the same chunks and embeddings
- **Hugging Face Inference API** — embeds each incoming question, no local model loaded
- **Groq** — generates answers, hosted, fast, free tier
- **Streamlit** — the interface, shared by both versions

---

## Directory Structure

```
company-report-assistant/
├── app.py                         # Streamlit interface, works with both architectures
├── pipeline/
│   ├── load_documents.py          # PDF text extraction
│   ├── chunk_and_embed.py         # Splitting and embedding (local)
│   ├── retrieve_answer.py         # Local retrieval and answer chain (Chroma + Ollama)
│   ├── retrieve_answer_cloud.py   # Cloud retrieval and answer chain (Pinecone + Groq)
│   ├── prompt_template.py         # Shared prompt, used by both versions
│   ├── migrate_to_pinecone.py     # One-time script: uploads local chunks to Pinecone
│   └── export_chunks_for_cloud.py # One-time script: exports chunk text for cloud BM25
├── data/
│   ├── *.pdf                      # Source annual reports (local use only)
│   └── chunks_export/             # Lightweight chunk text, used by the deployed version
├── eval/
│   └── qa_test_set.xlsx           # Manual evaluation set, 18 real Q&A pairs
├── diagnostics/                   # One-off scripts used to trace real retrieval bugs
├── vector_store/                  # Local Chroma store, generated, not tracked in git
├── Dockerfile                     # Builds the deployed container
├── requirements.txt               # Full local dev dependencies
├── requirements-cloud.txt         # Lean dependencies for the deployed container
├── README.md
└── KNOWLEDGE.md
```

---

## Limitations

Comparing companies against each other and uploading your own report are not supported. Retrieval accuracy is generally strong after two rounds of real, measured tuning (bigger embedding model, hybrid vector + keyword search, prompt and chunking improvements), but on broad questions asked against very large reports, the app can occasionally surface a different real section than expected, or blend multiple real facts inconsistently. All of this, including the real before/after evaluation numbers, is documented in [KNOWLEDGE.md](./KNOWLEDGE.md).
