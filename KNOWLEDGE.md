# KNOWLEDGE.md - Company Report Assistant

A living decision log. Every entry answers what was decided, why, when, what else was considered, and what the actual consequence was. Entries are not edited to look retroactively correct. When a decision changes, a new entry supersedes the old one, and the old reasoning stays visible.

---

## 1. Project Definition

Company Report Assistant answers questions about Indian companies using their real annual reports as the source of truth. A user picks a company, types a question in plain English, and gets an answer pulled directly from that company's actual report text, along with a way to check exactly where the answer came from. This is a RAG system (Retrieval Augmented Generation), meaning the chatbot searches the real document first and answers only from what it finds there, instead of answering from general knowledge.

Two working versions exist. **v1.0** runs entirely locally (Chroma, a local embedding model, Llama 3.2 via Ollama), free but not deployable on any free hosting tier. **v2.0** runs the same RAG pipeline but with the vector store, query embedding, and LLM each moved to a free hosted service (Pinecone, Hugging Face's Inference API, Groq), making it deployable for real. Both are documented below, in the order they happened.

---

## 2. Tech Stack + Why

### Why LangChain over LlamaIndex
**Considered:** LlamaIndex, another retrieval orchestration framework with strong document-indexing support.
**Chosen:** LangChain.
**Reasoning:** Both solve the same core problem. LangChain has broader adoption and stronger documentation for a full pipeline end to end, chunking through to the answer chain. Picked one framework to go deep in rather than splitting effort across two.

### Why Chroma for local development, Pinecone added later for deployment
**Considered (v1.0):** Pinecone and Weaviate, both hosted vector databases with more scaling features than needed here.
**Chosen (v1.0):** Chroma. Runs locally, free, no cloud account needed. A hosted database would have added infrastructure a 5-document project didn't need.
**Revised (v2.0):** Pinecone added specifically for the deployed version, once local memory measurements showed Chroma plus a local embedding model couldn't fit any free hosting tier (see the v2.0 deployment entry below). Local development still uses Chroma unchanged. This wasn't the original choice being wrong, it was a new, different requirement (deployability) that Chroma was never meant to solve.

### Why BAAI/bge-base-en-v1.5, over all-MiniLM-L6-v2 or OpenAI embeddings
**Considered:** OpenAI's embedding API. Started with `sentence-transformers/all-MiniLM-L6-v2`, a smaller, faster free model, as the first real choice.
**Chosen:** `BAAI/bge-base-en-v1.5`, a larger free model trained specifically for retrieval (matching short questions to longer passages), not general sentence similarity.
**Reasoning:** MiniLM worked at first, but the evaluation set surfaced a specific failure: it couldn't reliably tell "US business contributed 29% of revenue" apart from "India contributed 34% of revenue" elsewhere in the same report, structurally similar sentences the smaller model's limited resolving power confused. Confirmed by comparing raw retrieval rankings before and after the switch (see Discoveries): a chunk that was completely absent from the top 15 results moved to rank 4. OpenAI's API was ruled out to keep the project cost-free. The switch required a full vector store rebuild, several real hours on CPU-only hardware, accepted because it fixed a diagnosed problem, not a guessed one.

### Why chunk size changed from 1000/150 to 1300/200
**Considered, initially:** Smaller chunks (300-500 characters) for tighter matches, or larger chunks (2000+) for more surrounding context.
**Chosen, initially:** 1000 characters, 150 overlap. Sampled pages showed dense paragraphs, especially in the FMCG and Energy reports (3800+ characters per sampled page), and 1000 characters held one full idea without dilution.
**Revised to:** 1300 characters, 200 overlap, after the evaluation set surfaced a coherence problem (an answer that read like scrambled source text rather than a proper synthesis), on top of the ongoing infographic-page issue (see Discoveries). Larger chunks keep more of an idea together, reducing the chance a fact gets split at a boundary.
**Reasoning:** The original setting wasn't wrong, it was just never tested against real accuracy until the evaluation set existed. This was one of three changes made in the same round (with a prompt fix and a BM25 normalization fix). Retesting confirmed real improvement on the specific failing cases. The rebuild dropped total chunk count from 7,995 to 6,282 across all 5 companies, fewer, larger chunks, as expected.

### Why Ollama with Llama 3.2, over Anthropic or OpenAI's API
**Considered:** Claude and OpenAI's APIs, both strong, with Claude known for following "only use the provided context" instructions well. Also tried Llama 3.1 8B locally before switching down to 3.2 3B.
**Chosen:** Ollama, running Llama 3.2 3B locally.
**Reasoning:** Both paid APIs require ongoing cost, unnecessary for a project this size. Llama 3.1 8B worked but was slow enough on CPU-only hardware (no dedicated GPU, integrated Intel Iris Xe) to hurt the actual experience. Switching to the smaller 3.2 3B model cut response time with an acceptable depth trade-off. Honest cost: local models are slower and a step below Claude/GPT-4 class quality, acceptable here since the goal was a working, documented pipeline, not the highest-end answer quality.

### Why page citation is shown separately from the main answer
**Considered:** Putting the page number inline in the answer text itself.
**Chosen:** A short source line below the answer, plus a separate tab with the full retrieved chunk and page number.
**Reasoning:** Most readers don't need a page number inside the sentence, but removing it entirely would remove the main trust signal of a RAG system: that an answer is grounded and checkable. This keeps the answer clean while keeping verification one click away.

### v1.0: why this project was not deployed live
**Considered:** Docker on Render at increasing tiers as real memory was measured (free and $7/month, both 512MB, then $25/month at 2GB, then $85/month at 4GB). Streamlit Community Cloud, which had the same ~1GB ceiling and, more fundamentally, isn't built to run a background service like Ollama at all. Switching the deployed LLM to a free hosted API (Groq) to cut the LLM's memory cost specifically.
**Chosen:** No live deployment for v1.0.
**Reasoning:** Real memory usage was measured directly with Windows process monitoring, not estimated: roughly 941MB for the app itself (embeddings model, Chroma, Streamlit, LangChain) plus roughly 2GB more for a locally-run LLM. That ruled out every free tier available at the time. Even with the Groq swap, the app's own footprint alone still exceeded free-tier limits everywhere, meaning a paid tier (at least $25/month) would still have been required. Weighed against the ongoing cost, the decision was to stop and document this clearly rather than add more infrastructure changes under time pressure. This was evidence-based, not an unfinished task. See the v2.0 entry below for what changed.

### v2.0: making free deployment possible by moving storage and embedding off the container
**Considered:** Staying on v1.0's architecture with no live deployment, permanently. Paying for a bigger Render tier as a simpler, costlier fix.
**Chosen:** Keep the same RAG pipeline, LangChain orchestration, and evaluated retrieval logic, but move two specific pieces off the deployed container: the vector store to Pinecone (free tier), and query-time embedding to Hugging Face's hosted Inference API, alongside a Groq swap for the LLM. Local development is unaffected.
**Reasoning:** The v1.0 measurement showed the container needed ~941MB mostly from loading the embedding model and its PyTorch dependency, plus the vector index. Neither needs to live inside the app's own container: a vector database can be queried over a network, and a single short question can be embedded via an API call instead of a locally-loaded model. Pinecone's free tier (2GB storage, ~350,000 vectors) comfortably fits this project's real size (6,282 chunks), confirmed against Pinecone's actual current limits, not assumed (last verified August 2026). Combined with Groq, this removes the three heaviest pieces from the deployed app entirely. Nothing about the underlying engineering changes, this is a hosting change, not a rebuild.

### v2.0: decided to keep full hybrid search in the deployed version, not just vector search
**Considered:** Dropping BM25 keyword search for the deployed version and using Pinecone vector search alone, since Pinecone has no simple "give me every chunk for company X" the way Chroma does.
**Chosen:** Export each chunk's text and metadata (not the embedding vectors, which BM25 doesn't need) into small JSON files, one per company, shipped as part of the app's own code.
**Reasoning:** Total export size across all 5 companies is about 7.7MB, small enough to commit directly to the repository, smaller than the 85MB of source PDFs already tracked. This lets the deployed version build the same BM25 index instantly at startup with no network fetch, while Pinecone handles the vector half over the network. Keeps the deployed version's retrieval genuinely identical to local, not a simplified version of it.

### v2.0: why openai/gpt-oss-20b on Groq, not llama-3.1-8b-instant
**Considered:** `llama-3.1-8b-instant` on Groq, matching the local Llama model family.
**Chosen:** `openai/gpt-oss-20b`.
**Reasoning:** Groq deprecated `llama-3.1-8b-instant` in June 2026. Their own migration guidance pointed to `openai/gpt-oss-20b` as the direct replacement, still free, still fast, but an OpenAI open-weight model rather than Llama, a genuine if minor deviation from the original plan, worth stating plainly rather than glossing over. Local development is unaffected and still uses Llama 3.2 via Ollama. Last verified August 2026, Groq's available models can change again.

---

## 3. Data / System Overview

**Input data:** five Indian companies' latest annual reports, one per sector, chosen as sector leaders.

| Company | Sector | Pages | File size | PDF format |
|---|---|---|---|---|
| HDFC Bank | Banking and Finance | 629 | 17.7 MB | Text-based, no OCR needed |
| TCS | IT and Technology | 360 | 21.1 MB | Text-based, no OCR needed |
| HUL | FMCG | 244 | 29.2 MB | Text-based, no OCR needed |
| Reliance Industries | Energy and Oil and Gas | 147 | 9.2 MB | Text-based, no OCR needed |
| Sun Pharma | Pharma and Healthcare | 344 | 8.2 MB | Text-based, no OCR needed |

Checked with `pdfplumber` by sampling the first two pages and one middle page of each report. All five passed, none required OCR.

**v1.0 data flow (local):**
1. Each PDF is loaded and text extracted with `pdfplumber`.
2. Text is split into chunks with LangChain's `RecursiveCharacterTextSplitter`, 1300 characters, 200 overlap (revised from 1000/150, see Tech Stack).
3. Each chunk is embedded and stored in Chroma, tagged with company, sector, and page.
4. User selects a company, filtering retrieval to that company's chunks.
5. The question is embedded and compared against stored chunks for the closest matches.
6. Matched chunks and the question go to the LLM, which answers grounded only in that context.
7. The answer is shown with a short source line, full chunk detail available in a separate tab.

**v2.0 data flow (deployed), differences from v1.0 only:**
- Steps 1-3 happen once locally, then chunks (with their already-computed embeddings) are migrated to Pinecone via `migrate_to_pinecone.py`, and chunk text is separately exported to JSON via `export_chunks_for_cloud.py` for the deployed BM25 index.
- Step 5's embedding call goes to Hugging Face's Inference API instead of a locally-loaded model.
- Step 6's LLM call goes to Groq instead of Ollama.
- Everything else, including the hybrid search logic and the prompt, is identical.

**Early assumption worth flagging:** the pipeline assumes each report's text extracts into a reasonably coherent block per chunk. Reports with heavy tables or multi-column layouts extract messier than plain paragraphs. This turned out to be a real, confirmed issue, not a theoretical one, see Discoveries and Challenges.

---

## 4. Discoveries & Findings

**v1.0, initial build:**
- All 5 reports extracted as real, usable text, none needed OCR. Not guaranteed going in, Indian companies don't follow one standard reporting format.
- HDFC Bank's report is nearly double the page count of the next largest (629 vs. 360 for TCS), unanticipated when picking companies by sector leadership alone.
- Character density per page varies a lot by sector: HUL and Reliance averaged 3800+ characters per sampled page, HDFC and TCS closer to 1200-1500, part of why sector diversity was chosen for this project.
- Full extraction confirmed the sampled-page check was reliable: only 1 page out of 1,723 total had no extractable text (likely a cover or image-only page).
- The first end-to-end test question (HDFC Bank digital banking) returned a specific, accurate answer from 4 real pages, including a named product ("y Pixel" credit card) and a real statistic (90% of transactions digital), a strong early signal the grounding actually worked, not just in theory.
- Switching from Llama 3.1 8B to Llama 3.2 3B, plus reducing retrieved chunks from 4 to 3, brought real response time down to 14-18 seconds per question, a felt improvement confirmed through repeated use.

**v1.0, evaluation round 1:**
- Running the full 18-question evaluation set surfaced a consistent pattern: roughly a third of questions returned wrong or missing content, nearly all clustered around dense, infographic-style stat pages rather than regular prose. TCS's financial highlights page, for example, extracts as `"Revenue Growth Industry-leading EPS Growth ... 4.6% Operating Margin 25.0% 8.8%"`, a real sentence-shaped jumble caused by a 3-column layout flattening into one text stream. Narrative pages (Reliance's Jio section) retrieved and answered correctly; stat-grid pages (HDFC's key metrics, TCS's business model page, Sun Pharma's US business page) were the source of most misses, even when the answer was confirmed present by hand.
- A second, smaller pattern: on the largest reports, broad questions ("how does the bank manage risk") sometimes retrieved a real but different section than the one picked for the evaluation set, since these reports repeat themes across many sections. Still accurate, just not the expected passage.

**v1.0, embedding model upgrade:**
- Re-tested the Sun Pharma US business question after switching from `all-MiniLM-L6-v2` to `BAAI/bge-base-en-v1.5`. Before: the correct page (36) was absent from the top 15 vector results entirely. After: it appeared at rank 4, confirming the smaller model genuinely couldn't distinguish "US business" from similarly-worded "[other region] business" chunks elsewhere in the report. The app's actual answer afterward was correct (29%, ₹168.2 Billion), though the final 3 retrieved chunks were pages 37, 107, and 42, not page 36 itself, page 37 was a continuation of the same section and carried enough context. A real fix, just not through the exact mechanism predicted.

**v1.0, evaluation round 2 (post embedding upgrade + hybrid search):**
- Honest outcome: 4 questions changed from wrong/missing to correct (HDFC digital banking, HDFC emerging risks, TCS operating margin, Sun Pharma US revenue). But 2-3 questions got measurably worse on the same rerun (HDFC CSR/village figures lost a previously-correct number, TCS workforce answer got vaguer, HDFC AI customer-service answer became less coherent). A separate pattern on 4 more questions: the app gave a different real answer between runs, not wrong, just different valid content from a different section of the same large report, since a single hand-picked "expected answer" doesn't always capture every correct response a big report could give.

**v1.0, evaluation round 3 (prompt fix + BM25 normalization + chunk size increase):**
- Two clear, confirmed fixes: the garbled HDFC AI customer-service answer became clean and accurate, and the previously unanswerable Sun Pharma drug launches question now correctly names real drugs (Leqselvi, Unloxcyt) from the report's actual launch history.
- Two partial improvements: HDFC CSR correctly states the crore figure again and now honestly says it can't find the village count instead of inventing an unrelated one; HUL's manufacturing investment question moved from a flat "could not find" to genuinely relevant real content, just not the exact figure expected.
- One unresolved case: Reliance's solar energy plans question produced an internally inconsistent answer this round, citing two capacity targets for two different years that don't logically fit together, suggesting the model blends multiple real, separate facts rather than picking one coherent thread.

**v2.0, deployment redesign:**
- Hugging Face's hosted Inference API confirmed working for `BAAI/bge-base-en-v1.5` via a direct test call before wiring it in: returned a 768-dimension embedding, matching the local model and Pinecone's index exactly. Resolved the real uncertainty about free-tier availability by testing, not assuming.
- All 6,282 chunks migrated to Pinecone and independently verified: Pinecone's own `describe_index_stats()` call confirmed 6,282 total vectors, matching the local Chroma count exactly.
- The cloud retrieval path, tested on the same Sun Pharma question, retrieved the exact target page (36) directly, alongside real adjacent context, genuinely matching or exceeding the local hybrid search's result on this specific case.
- Measured container memory with `docker stats` after asking a real question: **185.3MB total**, versus v1.0's measured ~941MB app plus ~2GB for a local LLM, a real, verified reduction of roughly 94%.
- Deployed live on Render's free tier. Build succeeded on the first attempt. Real answers generate in under 10 seconds, faster than local Ollama's 14-18 seconds, Groq's inference speed advantage showing in practice, not just on paper.

---

## 5. Challenges & How They Were Solved

### LangChain text splitter import error
**Problem:** `chunk_and_embed.py` failed immediately with `ModuleNotFoundError: No module named 'langchain.text_splitter'`.
**Investigation:** `RecursiveCharacterTextSplitter` used to live in core `langchain`. Checked the installed version and found text splitters had moved to a separate package, `langchain-text-splitters`, in a recent restructuring.
**Fix:** Installed `langchain-text-splitters`, updated the import, added the package to `requirements.txt`.
**Why this fix, not another:** Pinning an older `langchain` version would avoid the error but lose newer fixes. Updating the import to match the library's current structure is the durable fix.

### Missing torchvision dependency
**Problem:** `app.py` failed with `ModuleNotFoundError: No module named 'torchvision'`, from inside the `sentence-transformers`/`transformers` import chain, not from project code.
**Investigation:** Some `transformers` versions import `torchvision` unconditionally at load time to support image features, even though this project only does text embeddings.
**Fix:** Installed `torchvision` directly, added it to `requirements.txt`.
**Why this fix, not another:** Pinning an older `transformers` version risks losing other fixes with no long-term stability guarantee. Installing the one missing dependency is simpler and more durable, even though the package itself is never actually used here.

### Slow response time with Llama 3.1
**Problem:** Once the pipeline worked end to end, Llama 3.1 8B answers took long enough to hurt the actual experience of using the app.
**Investigation:** Checked for a dedicated GPU. None (integrated Intel Iris Xe only), meaning CPU-only inference for an 8B model, inherently slow, not a code bug.
**Fix:** Switched to Llama 3.2 3B, reduced retrieved chunks from 4 to 3, added streaming so text appears as it generates instead of after one long wait.
**Why this fix, not another:** For a tool meant to be tried by other people, response time matters as much as depth. A faster, slightly shallower model people will actually enjoy using beats a deeper one that feels broken. Llama 3.1 was removed entirely once the switch was confirmed good.

### EnsembleRetriever import path, again
**Problem:** While building hybrid search, `from langchain.retrievers import EnsembleRetriever` failed with the same category of error as the text splitter issue.
**Investigation:** Tested imports directly rather than guessing. Found `EnsembleRetriever` now lives in a separate package, `langchain-classic`. Second time in this project a LangChain import moved between packages mid-build, a real pattern in this dependency's restructuring, not a one-off.
**Fix:** Installed `langchain-classic`, updated the import.
**Why this fix, not another:** Same reasoning as the text splitter fix. Worth noting for future work: checking the current package location directly, rather than assuming the old path, is now the expected first step if another LangChain import breaks.

### No progress feedback during a genuinely long embedding run
**Problem:** After switching to the bigger `BAAI/bge-base-en-v1.5` model, `chunk_and_embed.py` gave no visible progress, just silence for over 30 minutes with no way to tell if it was working or stuck.
**Investigation:** Confirmed via Task Manager the process was genuinely active (58% CPU), not frozen. `Chroma.from_documents()` embeds the entire chunk list in one call with no built-in progress reporting, and the script's original time estimate ("a few minutes") was written for the smaller, faster MiniLM model.
**Fix:** Rewrote `embed_and_store` to process chunks in batches of 200 using `add_documents()`, printing a running count and percentage. Corrected the printed estimate to the real range (30-60+ minutes on CPU-only hardware).
**Why this fix, not another:** For a script other people might run themselves, sitting silent for potentially an hour with zero indication of progress is a real usability problem, not a personal inconvenience. Batching also means an interrupted run keeps its partial progress instead of losing everything.

### Chroma embeddings not JSON-serializable for Pinecone upload
**Problem:** `migrate_to_pinecone.py` failed with `TypeError: Type is not JSON serializable: numpy.float64` on the first upload attempt.
**Investigation:** Chroma returns stored embeddings as NumPy numbers. Pinecone's upload encoder can only serialize plain Python types.
**Fix:** Explicitly converted each embedding value to a plain Python `float` before upload.
**Why this fix, not another:** A one-line, low-risk fix at the exact point of failure, no reason to look further.

### Pinecone metadata returns numeric page numbers as floats
**Problem:** Page citations from the cloud retrieval path displayed as `37.0` instead of `37`, while the local BM25 path (from the exported JSON) displayed correctly as `37`.
**Investigation:** Pinecone stores and returns numeric metadata as floats regardless of how it was uploaded, an internal storage detail, not a bug in this project's code.
**Fix:** Added a small `clean_page_number()` helper that normalizes either case to a clean int before display.
**Why this fix, not another:** Small, targeted fix at the display layer, cheaper and more robust than trying to control Pinecone's internal storage format.

---

## 6. Limitations

- **Cross-company comparison is not supported.** The five companies span different sectors on purpose, so their reports don't share comparable metrics. Comparing a bank against an FMCG company head to head wouldn't produce a meaningful answer.
- **User-uploaded reports are not supported.** Only the five pre-loaded companies can be queried. A reasonably scoped next feature, left out to get the core pipeline working first.
- **No automatic lookup by typing any company name.** No reliable, standard way exists to locate any given Indian company's latest annual report from just its name. Considered and rejected as its own separate project, not a small addition here.
- **Retrieval is noticeably weaker on infographic-style stat pages than on regular prose.** About a third of evaluation questions were affected in some way, either returning no answer for content that genuinely exists, or a real but differently-sourced answer. A known, common limitation of text-only PDF extraction for RAG, not unique to this project's code. A real fix would need page-layout-aware chunking or table-specific extraction, left as a future improvement.
- **Fixing retrieval on one question can make a different question worse.** Real, measured gains on some questions came alongside real regressions on others in the same rerun. A follow-up fix later resolved one specific regression (answer coherence), but the broader trade-off is not fully solved: changing what gets retrieved for one question can shift what gets retrieved for a nearby one, not always for the better.
- **Answers to broad questions on large reports are not always consistent between runs.** The same broad theme is often discussed in more than one place in a 300+ page report, so the app can surface different real sections on different runs. Generally still accurate to some real part of the report, just not always the specific passage expected.
- **On some broad questions, the model blends multiple real facts inconsistently.** Reliance's solar energy plans question produced three different answers across three test rounds, and once cited two capacity targets for two different years that don't logically fit together. Unresolved. A deeper fix (teaching the model to notice and flag conflicting retrieved content instead of blending it silently) is a real future idea, not attempted here.
- **The deployed (v2.0) version depends on three external free services staying available and free.** Pinecone, Hugging Face's Inference API, and Groq could each change their free-tier terms or model availability (Groq already deprecated one model mid-project, see Tech Stack). Local development (v1.0) has no such dependency. Last verified working end to end: August 2026.

---

## 7. Changelog

Chronological index of major milestones. Full reasoning for each lives in the sections above, this list points to it rather than repeating it.

- **Planning:** Project scoped, 5 companies picked, PDFs checked for text quality.
- **`load_documents.py`:** Built and tested, 1,723 pages extracted across all 5 PDFs.
- **`chunk_and_embed.py`:** Built and tested, 7,995 chunks at 1000/150 settings. Hit and fixed the LangChain text splitter import issue (§5).
- **`retrieve_answer.py`:** Built and tested, first accurate answer confirmed end to end.
- **Evaluation set built:** 18 hand-verified questions across all 5 companies, real answers pulled from source PDFs.
- **Evaluation round 1:** Roughly a third of questions failed, root cause traced to infographic-style pages (§4).
- **v1.0 pushed to GitHub.**
- **Hybrid search built:** BM25 + vector search combined via `EnsembleRetriever`. Hit the `langchain-classic` import issue (§5).
- **Embedding model upgraded** to `BAAI/bge-base-en-v1.5` (§2, §4).
- **Evaluation round 2:** Real gains and real regressions from the model upgrade (§4).
- **Third tuning round:** Prompt fix, BM25 normalization, chunk size increase to 1300/200 (§2). Hit the silent-progress issue during re-embedding (§5).
- **Evaluation round 3:** Two confirmed fixes, one unresolved case remaining (§4).
- **v1.0 deployment investigated, not pursued:** Real memory measured, no free tier fit (§2).
- **v2.0 started:** New architecture scoped to make free deployment possible (§2).
- **Chunks migrated to Pinecone:** 6,282 verified. Hit the NumPy serialization issue (§5).
- **Hugging Face Inference API confirmed working** for query embedding (§4).
- **Hybrid search preserved for the deployed version** via exported chunk JSON (§2).
- **Cloud retrieval built and verified**, correctly retrieved the previously-hard Sun Pharma case (§4). Hit the Pinecone float page-number issue (§5).
- **Groq wired in**, full cloud pipeline confirmed end to end (§2).
- **Dockerfile built, containerized, memory measured:** 185.3MB, a 94% reduction from v1.0 (§4).
- **v2.0 deployed live on Render's free tier.** Answers generate in under 10 seconds (§4).

---

## Still Open

- BM25's literal "%" vs "percentage" keyword mismatch was normalized (fixed in the third tuning round), but query expansion or synonym handling more broadly is not implemented. Not urgent, the embedding model does most of the real work in the hybrid combination now.
- Reliance's solar plans inconsistency (§6) is unresolved.
- The infographic-page retrieval weakness (§6) is a known, accepted limitation, not planned to be fixed in the current scope.
