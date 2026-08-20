# KNOWLEDGE.md — Company Report Assistant

A living decision log, updated as the project happens, not after.

---

## 1. Project Definition

Company Report Assistant is a chatbot that answers questions about Indian companies using their real annual reports as the source of truth. A user picks a company from a dropdown, types a question in plain English, and gets an answer pulled directly from that company's actual report text, along with a way to check exactly where the answer came from. This uses an approach called **RAG**, short for Retrieval Augmented Generation, meaning the chatbot searches the real document first and answers only from what it finds there, instead of answering from general knowledge.

---

## 2. Tech Stack + Why

### Why LangChain over LlamaIndex
**Considered:** LlamaIndex, another retrieval orchestration framework with strong document-indexing support.
**Chosen:** LangChain.
**Reasoning:** Both frameworks solve the same core problem. LangChain has broader adoption and stronger community documentation for building a full pipeline end to end, from chunking through to the final answer chain. Picked one framework to go deep in, rather than splitting effort learning two shallowly.

### Why Chroma over Pinecone or Weaviate
**Considered:** Pinecone and Weaviate, both hosted vector databases with more scaling features.
**Chosen:** Chroma.
**Reasoning:** This project works with 5 documents, not thousands. Chroma runs locally, is free, and needs no cloud account to get started. A hosted vector database would add infrastructure this project does not need at its current size. Might revisit if the scale changes later.

### Why Hugging Face sentence-transformers or OpenAI for embeddings
**Considered:** OpenAI's embedding API as the default, since it is well documented and commonly used.
**Chosen:** Testing both, final pick pending.
**Reasoning:** A free Hugging Face model avoids ongoing API cost for a project this size, but OpenAI's embeddings may perform better on dense financial text. Both will be tested against the same document set before locking a final choice. This entry will be updated once that test is done, with actual results, not just the plan.

### Why Docker and Render for deployment
**Considered:** Deploying without containerization, directly on a hosting platform.
**Chosen:** Docker, deployed on Render.
**Reasoning:** Docker means the app runs the same way regardless of where it is hosted, which matters once the project needs to move between environments or be reproduced elsewhere. Render supports the kind of always-on server this chat app needs, unlike platforms built for short-lived requests.

### Why chunk size 1000 characters with 150 character overlap
**Considered:** Smaller chunks (around 300 to 500 characters) for tighter, more precise matches. Larger chunks (around 2000+ characters) to preserve more surrounding context per match.
**Chosen:** 1000 characters, 150 character overlap, using LangChain's `RecursiveCharacterTextSplitter`.
**Reasoning:** Sampled pages from the actual reports showed dense paragraph writing, particularly in the FMCG and Energy sector reports, both averaging over 3800 characters of extractable text per sampled page. A 1000 character chunk generally holds one full idea without diluting it across too much unrelated text. The 150 character overlap is meant to stop a sentence from being cut in half exactly at a chunk boundary. This is a starting point, not fixed. It will be revisited if retrieved answers come back vague or miss obvious content once real queries are tested.

### Why Ollama with Llama 3.2, over Anthropic or OpenAI's API
**Considered:** Anthropic's Claude API and OpenAI's GPT API, both strong options for the final answer generation step, with Claude in particular known for following strict "only use the provided context" instructions well. Also considered Llama 3.1 8B as the local model before switching to the smaller Llama 3.2 3B.
**Chosen:** Ollama, running Llama 3.2 3B locally.
**Reasoning:** Both Claude and OpenAI require adding paid credit to an API account, no free tier for ongoing use. For a portfolio project of this size, spending money was not necessary when a genuinely capable free option exists. Ollama runs locally at no cost, with no API key and no usage limit. Llama 3.1 8B was tried first and worked correctly, but response time on this machine's CPU (no dedicated GPU, integrated Intel Iris Xe graphics only) was slow enough to hurt the actual experience of using the app. Switching to Llama 3.2 3B, a smaller model, cut response time noticeably with an acceptable drop in answer depth. The real trade-off, documented honestly, is that local models are slower to respond than a paid API and answer quality is a step below Claude or GPT-4 class models, and the smaller 3B model trades a bit more quality for speed compared to the 8B version. That trade-off is acceptable here, since the goal of this project is a working, well-documented RAG pipeline, not the fastest or highest-end possible answer quality.

### Why page citation is shown separately from the main answer, not inline
**Considered:** Putting the page number directly inside the answer text itself, for example ending every answer with "(page 47)".
**Chosen:** A short source line below the answer, plus a separate tab with the full retrieved chunk and page number.
**Reasoning:** Most people reading an answer do not need a page number inside the sentence itself. But removing it completely would remove the main trust signal of a RAG system, that an answer is grounded in a real, checkable document. The middle ground keeps the main answer clean while keeping verification easy to reach.

---

## 3. Data / System Overview

**Input data:** Five Indian companies' latest annual reports, one company per sector, chosen as sector leaders:

| Company | Sector | Pages | File size | PDF format |
|---|---|---|---|---|
| HDFC Bank | Banking and Finance | 629 | 17.7 MB | Text-based, no OCR needed |
| TCS | IT and Technology | 360 | 21.1 MB | Text-based, no OCR needed |
| HUL | FMCG | 244 | 29.2 MB | Text-based, no OCR needed |
| Reliance Industries | Energy and Oil and Gas | 147 | 9.2 MB | Text-based, no OCR needed |
| Sun Pharma | Pharma and Healthcare | 344 | 8.2 MB | Text-based, no OCR needed |

Checked using `pdfplumber` by sampling the first two pages and one middle page of each report and confirming extractable text was present. All five passed, none require OCR.

**Data flow (planned):**
1. Each PDF is loaded and text is extracted with `pdfplumber`.
2. Text is split into chunks using LangChain's `RecursiveCharacterTextSplitter`, 1000 characters with 150 character overlap.
3. Each chunk is converted into an embedding and stored in Chroma, tagged with company name and page number.
4. User selects a company in the dropdown, which filters retrieval to only that company's stored chunks.
5. User's question is embedded and compared against stored chunks to find the closest matches.
6. Matched chunks and the question are sent to the LLM, which generates an answer grounded only in those chunks.
7. Answer is shown with a short source line, full chunk detail available in a separate tab.

**Early assumption worth flagging:** the pipeline assumes each report's text extracts cleanly enough that a 1000 character chunk lands on a reasonably coherent block of text. Reports with heavy tables or multi-column layouts may extract messier than plain paragraph text does. This has not been tested yet on the full documents, only sampled pages, and is a likely source of early bugs.

---

## 4. Discoveries & Findings

- All 5 reports came back as real, extractable text on the sampled pages, none needed OCR. This was not guaranteed going in, PDF quality was expected to vary more given that Indian companies do not follow one standard reporting format.
- HDFC Bank's report is close to double the page count of the next largest report (629 pages versus 360 for TCS). This size difference was not anticipated when the 5 companies were picked based on sector leadership alone, and will need to be handled cleanly by the chunking step rather than assumed away.
- Character density per page varies a lot by sector. HUL and Reliance Industries averaged over 3800 characters per sampled page, while HDFC Bank and TCS averaged closer to 1200 to 1500. This suggests report writing style differs meaningfully by sector, which is part of why sector diversity was chosen for this project in the first place.
- Full extraction across all 5 PDFs confirmed the sampled-page check was reliable. Only **1 page out of 1,723 total pages** across all reports had no extractable text, from HDFC Bank's report, likely a cover page or an image-only page. Extraction otherwise worked cleanly on every page, no OCR needed anywhere.
- Switching from Llama 3.1 8B to Llama 3.2 3B, combined with reducing retrieved chunks from 4 to 3, brought real answer generation time down to roughly **14 to 18 seconds** per question on this machine (CPU only, no dedicated GPU). This is a genuine, felt improvement over the 8B model, confirmed through actual repeated use, not just a theoretical expectation from picking a smaller model.
- Running the full 18-question evaluation set surfaced a real, consistent pattern in where the system struggles. Roughly a third of questions returned wrong or missing content, and nearly all of those failures cluster around one specific type of source page: dense, infographic-style stat pages, rather than regular prose. For example, TCS's own financial highlights page extracts as `"Revenue Growth Industry-leading EPS Growth ... 4.6% Operating Margin 25.0% 8.8%"`, a real sentence-shaped jumble caused by a 3-column visual layout being flattened into one text stream during extraction. Numbers and their labels get visually scrambled in a way that plain prose never does. Pages with real narrative writing (for example, Reliance's Jio financial performance section) retrieved correctly and answered accurately. Pages built as stat grids or dashboards (HDFC's key metrics page, TCS's integrated business model page, Sun Pharma's US business overview page) were the source of most retrieval misses, even though the actual answer text was confirmed, by hand, to be present on those pages.
- A second, smaller pattern: on the largest reports (HDFC Bank, TCS, Reliance), broad questions like "how does the bank manage risk" sometimes retrieved a real but different section than the one originally picked for the evaluation set, since these large reports repeat similar themes (risk, digital strategy, partnerships) across many separate sections. The retrieved content was often still real and accurate, just not the specific passage expected, which matters less for the app's usefulness than a true wrong answer would, but is worth being aware of.
- Chunking at 1000 characters with 150 character overlap produced **7,995 total chunks** across all 5 reports. Sun Pharma alone produced the largest jump in chunk count relative to its page count, likely due to denser paragraph text per page compared to the others, worth watching once real question testing begins.
- First end-to-end test question, asked about HDFC Bank's digital banking initiatives, returned a specific, accurate answer pulled from 4 different pages (112, 90, 155, 233), including specific details like a named product ("y Pixel" credit card) and a real statistic (90 percent of transactions happening digitally). This is a strong early signal that the retrieval and grounding setup works correctly on real data, not just in theory.

---

## 5. Challenges & How You Solved Them

### LangChain text splitter import error
**Problem:** Running `chunk_and_embed.py` for the first time failed immediately with `ModuleNotFoundError: No module named 'langchain.text_splitter'`.

**Investigation:** The code imported `RecursiveCharacterTextSplitter` from `langchain.text_splitter`, which used to be the correct location in older LangChain versions. Checked the installed LangChain version and found that text splitters had been moved out of the core `langchain` package into a separate package, `langchain-text-splitters`, in a recent restructuring.

**Fix:** Installed `langchain-text-splitters` directly with `pip install langchain-text-splitters`, and changed the import to `from langchain_text_splitters import RecursiveCharacterTextSplitter`. Added the package to `requirements.txt` so this does not break again on a fresh install.

**Why this fix, not another:** Could have pinned an older version of `langchain` that still had the old import path, but that would mean giving up newer fixes and features for the sake of one import line. Updating the import to match the current package structure is the correct long term fix, not a workaround.

### Missing torchvision dependency
**Problem:** Running `app.py` for the first time failed with `ModuleNotFoundError: No module named 'torchvision'`, coming from inside the `sentence-transformers` / `transformers` import chain, not from any code written directly for this project.

**Investigation:** Some versions of the `transformers` library, a dependency of `sentence-transformers`, attempt to import `torchvision` at load time to support image-processing features. This project only uses text embeddings, never touches image processing, but the import still runs unconditionally in the installed version, so a missing `torchvision` package crashes the whole import chain rather than being skipped.

**Fix:** Installed `torchvision` directly with `pip install torchvision`, and added it to `requirements.txt`.

**Why this fix, not another:** Could have tried pinning an older `transformers` version without this import behavior, but that risks losing other fixes and compatibility improvements, and isn't guaranteed to be stable long term. Installing the one missing dependency directly is the simpler, more durable fix, even though the package itself is not used for anything in this project beyond satisfying that internal import.

### Slow response time with Llama 3.1
**Problem:** Once the full pipeline was working end to end, answers from Llama 3.1 8B took a long time to generate, long enough to hurt the actual experience of using the app.

**Investigation:** Checked whether the machine had a dedicated GPU Ollama could use for acceleration. It did not, only integrated Intel Iris Xe graphics, meaning the model was running entirely on CPU. CPU inference for an 8 billion parameter model is inherently slow, this was not a bug in the code, just the real cost of running a model this size without GPU acceleration.

**Fix:** Switched to Llama 3.2 3B, a smaller model, which noticeably cut response time. Also reduced the number of retrieved chunks per question from 4 to 3, and added streaming so the answer appears word by word as it generates instead of all at once after a long wait, which does not reduce total generation time but removes the feeling that the app has frozen.

**Why this fix, not another:** Considered keeping Llama 3.1 and just accepting the wait, but for a tool meant to be tried and demoed by other people, response time matters as much as answer quality. A faster, slightly less deep model that people will actually enjoy using is a better outcome than a deeper model that feels broken. Once confirmed working well, Llama 3.1 was removed from the project and from the machine entirely, no reason to keep two models installed once the switch was confirmed good.

---

## 6. Limitations

- **Cross-company comparison is not supported.** The five companies span different sectors on purpose, so their reports do not share comparable metrics or risk categories. A comparison feature was considered and set aside because comparing a bank against an FMCG company head to head would not produce a meaningful answer.
- **User-uploaded reports are not supported in v1.** Only the five pre-loaded companies can be queried. Uploading your own report is a reasonably scoped next feature, left out of v1 to get the core dropdown-based pipeline working and tested first.
- **No automatic lookup by typing any company name.** There is no reliable, standard way to locate any given Indian company's latest annual report from just its name, since these are not published at predictable web addresses or through a shared system. This was considered and rejected as its own separate project, not a small addition here.
- **Embeddings and chunk size are not finalized.** Both are working starting points based on reasoning, not yet validated against real question and answer testing. Expect these settings to change once evaluation begins.
- **Retrieval is noticeably weaker on infographic-style stat pages than on regular prose.** Running the full evaluation set (see Discoveries above) showed that pages built as visual stat grids or dashboards, rather than written paragraphs, extract into a jumbled, hard-to-embed text order, which hurts the system's ability to find and use them correctly. Around a third of evaluation questions were affected by this in some way, either returning no answer for content that genuinely exists, or returning a real but differently-sourced answer than intended. This is a known, common limitation of text-only PDF extraction for RAG, not something unique to this project's code, and a fix would likely involve smarter page-layout-aware chunking or table-specific extraction, left as a future improvement rather than solved in v1.

---

## 7. Changelog / Timeline

- **Planning stage:** Project scoped, name finalized as Company Report Assistant, tech stack decided, 5 companies picked and their PDFs checked for text quality. Chunk size and overlap decided as a starting point. No code written yet.
- **`load_documents.py` built and tested:** Extracts text page by page from all 5 PDFs using `pdfplumber`, keeping page numbers attached for later citations. Ran successfully across all 5 reports, 1,723 total pages, only 1 page skipped for having no extractable text. Next step is chunking and embedding.
- **`chunk_and_embed.py` built and tested:** Splits each page's text into 1000 character chunks with 150 character overlap, embeds them using the free Hugging Face model `all-MiniLM-L6-v2`, and stores everything in Chroma with company, sector, and page metadata attached. Hit and fixed a LangChain import path issue along the way (see Challenges section). Final run produced 7,995 chunks across all 5 companies, stored successfully in the local vector store. Next step is building the retrieval and answer chain.
- **`retrieve_answer.py` built and tested:** Filters retrieval to a single company using Chroma's metadata filter, builds a prompt that restricts the LLM to only the retrieved context, and generates answers using Llama 3.1 running locally through Ollama, chosen over Anthropic or OpenAI's paid APIs (see Tech Stack section). First real test question against HDFC Bank's report returned a specific, accurate, correctly cited answer. Core RAG pipeline is now working end to end. Next step is building the Streamlit interface.
- **Evaluation set built:** 18 question and expected-answer pairs written across all 5 companies (6 for HDFC Bank, 3 each for the rest, weighted since HDFC Bank's report is by far the largest), stored in `eval/qa_test_set.xlsx`. Every expected answer was pulled directly from the real report text, not guessed, with the exact source page(s) noted for each.
- **Evaluation run and results reviewed:** All 18 questions asked in the running app, actual answers compared against expected answers. Roughly a third of questions returned wrong or missing content, with failures clustering specifically around infographic-style stat pages rather than regular prose (see Discoveries and Limitations above). Root cause identified and documented, not yet fixed. Hybrid search identified as the concrete next step to address this (see Still Open below).
- **v1 pushed to GitHub.** Core pipeline (load, chunk and embed, retrieve and answer), Streamlit interface, and evaluation set are all complete and working. Known retrieval accuracy issue on infographic-style pages is documented, not hidden, with a specific planned fix (hybrid search) rather than left as a vague "known issue."

---

## Still open

- **Hybrid search (vector + keyword/BM25 retrieval).** Evaluation showed that infographic-style stat pages sometimes rank low in pure vector similarity search because their scrambled text extraction produces a weak embedding, even when the literal answer is present on the page. Combining vector search with a keyword-based method like BM25, so a chunk can be retrieved either because it means something similar or because it literally contains the right words, is the identified next step to address this. Estimated as a real, multi-hour addition (new retriever setup, saving per-company chunk lists outside Chroma, re-testing against the evaluation set), not attempted yet, deliberately scheduled for after the first version was pushed and working.
- Final choice between Hugging Face and OpenAI embeddings, pending a side by side test
- Chunk size and overlap settings, may be revisited once hybrid search is in place and its own impact on accuracy is known
