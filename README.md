# Company Report Assistant

Answers natural-language questions about Indian companies using their real annual reports as the source, with every answer grounded in the actual document text and citable down to the page.

![Status](https://img.shields.io/badge/status-live-brightgreen)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B)
![LangChain](https://img.shields.io/badge/langchain-hybrid%20RAG-1C3C3C)
![Deployment](https://img.shields.io/badge/deployed-Render%20%2B%20Docker-46E3B7)

---

## What it does, and why it exists

Reading a 150-page annual report to find one answer is slow. This app lets you pick a company, ask a direct question, and get an answer pulled from the actual report, with a way to check exactly where it came from. It uses RAG (Retrieval Augmented Generation), so answers come from the real document, not the model's general knowledge, and it won't guess when the answer isn't in the report.

**🔗 Live demo:** [company-report-assistant.onrender.com](https://company-report-assistant.onrender.com)
*(Runs on a free hosting tier that spins down after inactivity. First request after idle time can take 30-60 seconds to wake up, after that, answers generate in under 10 seconds.)*

**Highlights:**
- 5 Indian companies across 5 different sectors: Banking, IT, FMCG, Energy, Pharma
- Hybrid retrieval (vector + keyword search), not vector search alone
- Answers stream in live and cite the exact source page
- Light and dark mode, no technical setup needed to use the live demo
- Two complete, working architectures: a free hosted deployment (Pinecone, Hugging Face, Groq) and a fully offline local setup (Chroma, Ollama), see [KNOWLEDGE.md](./KNOWLEDGE.md) for why both exist

---

## Installation

Want to just try it? Use the live demo linked above, nothing to install.

To run it locally instead (a separate, fully offline stack):

```bash
git clone https://github.com/khushi1215/company-report-assistant.git
cd company-report-assistant
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

Local development uses a free, local LLM, no API key or paid account needed. Install [Ollama](https://ollama.com), then pull the model:

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

Two complete architectures, same RAG pipeline and LangChain orchestration underneath, only where the heavy pieces run differs. Full reasoning, including real measured memory numbers, is in [KNOWLEDGE.md](./KNOWLEDGE.md).

**Local development:**
- **LangChain** — orchestrates retrieval and the answer chain
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

Comparing companies against each other and uploading your own report are not supported. Retrieval accuracy is generally strong after multiple rounds of real, measured tuning, but on broad questions asked against very large reports, the app can occasionally surface a different real section than expected, or blend multiple real facts inconsistently. Full details, including real before/after evaluation numbers, are in [KNOWLEDGE.md](./KNOWLEDGE.md).

---

## Contributing

This is a personal portfolio project, not actively seeking contributions. Feedback and questions are welcome via GitHub issues.

---

## License

No license file yet. All rights reserved for now, this is a personal project built for learning and portfolio purposes.
