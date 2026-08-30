"""
prompt_template.py

The prompt used to instruct the LLM, shared between the local
(retrieve_answer.py) and cloud (retrieve_answer_cloud.py) pipelines.
Kept in its own small file with no other dependencies, so importing
it never accidentally drags in heavier local-only libraries (like
the embedding model classes) through a hidden import chain.
"""

PROMPT_TEMPLATE = """You are answering questions using only the context provided below, taken from {company}'s annual report. Do not use any outside knowledge. If the answer is not clearly in the context, say you could not find that information in the report, do not guess.

Write your answer in clear, complete sentences of your own. The context below may be messy, extracted from tables, stat boxes, or multi-column page layouts, so it may not read like normal prose. Do not copy fragments or phrases directly from the context as-is. Read it, understand what it says, and explain it plainly in your own words instead.

Context from the report:
{context}

Question: {question}

Answer:"""
