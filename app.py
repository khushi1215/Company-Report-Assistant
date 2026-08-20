"""
app.py

Streamlit interface for Company Report Assistant.

Layout:
- Top header: title, theme toggle, company selector, a snapshot
  card (sector + page count), and a few suggested starter
  questions for the selected company. All in the main page,
  no collapsible sidebar.
- Below that, two tabs: "Ask" for the question/answer flow, and
  "Sources" for the full retrieved chunk text with page citations
  styled as bookmark tabs, echoing how someone would physically
  flag a page in a printed report.
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "pipeline"))

from load_documents import COMPANIES
from retrieve_answer import load_vector_store, retrieve_chunks, build_context_string, PROMPT_TEMPLATE
from langchain_ollama import ChatOllama


# ---------------------------------------------------------------
# Static metadata for the snapshot card. Page counts are the real
# numbers confirmed during the load_documents.py test run. Hardcoded
# here rather than re-reading every PDF on every app load, since
# these values don't change unless the source reports are replaced.
# ---------------------------------------------------------------
COMPANY_META = {
    "HDFC Bank": {"pages": 628, "sector_key": "banking"},
    "TCS": {"pages": 360, "sector_key": "it"},
    "HUL": {"pages": 244, "sector_key": "fmcg"},
    "Reliance Industries": {"pages": 147, "sector_key": "energy"},
    "Sun Pharma": {"pages": 344, "sector_key": "pharma"},
}

SECTOR_COLORS = {
    "banking": "#2F5D8A",
    "it": "#2F8A79",
    "fmcg": "#7A4B6E",
    "energy": "#B5651D",
    "pharma": "#3F7D5C",
}

SUGGESTED_QUESTIONS = {
    "HDFC Bank": [
        "What did the bank say about digital banking initiatives?",
        "What risks did the bank highlight this year?",
        "How did the bank describe its asset quality?",
    ],
    "TCS": [
        "What did the company say about AI and automation?",
        "What new client wins or deals were mentioned?",
        "What risks did the company highlight this year?",
    ],
    "HUL": [
        "What did the company say about rural demand?",
        "How did the company describe its new product launches?",
        "What sustainability initiatives were mentioned?",
    ],
    "Reliance Industries": [
        "What did the company say about its retail business?",
        "How did the company describe its energy transition plans?",
        "What risks did the company highlight this year?",
    ],
    "Sun Pharma": [
        "What did the company say about its US business?",
        "What new drug approvals or launches were mentioned?",
        "What risks did the company highlight this year?",
    ],
}


# ---------------------------------------------------------------
# Cached resources, loaded once per session, not on every rerun.
# ---------------------------------------------------------------
@st.cache_resource
def get_vector_store():
    return load_vector_store()


@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.2")


# ---------------------------------------------------------------
# Page setup and theme
# ---------------------------------------------------------------
st.set_page_config(page_title="Company Report Assistant", page_icon="📑", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

theme = "dark" if st.session_state["dark_mode"] else "light"

THEMES = {
    "light": {
        "ink": "#142433",
        "muted": "#4A5A66",
        "paper": "#EEF1F0",
        "surface": "#FFFFFF",
        "gold": "#A9812F",
        "line": "#C9CFCB",
        "input_bg": "#FFFFFF",
    },
    "dark": {
        "ink": "#ECEFF2",
        "muted": "#A6B0B8",
        "paper": "#10161C",
        "surface": "#1B242C",
        "gold": "#D9AE55",
        "line": "#313C46",
        "input_bg": "#1B242C",
    },
}

t = THEMES[theme]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

.stApp {{
    background-color: {t['paper']};
}}

h1, h2, h3, h4 {{
    font-family: 'Source Serif 4', serif;
    color: {t['ink']} !important;
}}

p, div, span, label, li {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {t['ink']};
}}

[data-testid="stAppViewContainer"] * {{
    color: {t['ink']};
}}

[data-testid="stTextInput"] input {{
    background-color: {t['input_bg']} !important;
    color: {t['ink']} !important;
    border: 1px solid {t['line']} !important;
}}

[data-testid="stRadio"] label {{
    background-color: {t['surface']};
    border: 1px solid {t['line']};
    border-radius: 20px;
    padding: 6px 16px;
    margin-right: 8px;
    color: {t['ink']} !important;
}}

[data-testid="stRadio"] label:has(input:checked) {{
    background-color: {t['ink']};
    border-color: {t['ink']};
}}

[data-testid="stRadio"] label:has(input:checked) p {{
    color: {t['paper']} !important;
}}

[data-testid="stRadio"] input {{
    display: none;
}}

hr {{
    border: none !important;
    border-top: 1px solid {t['line']} !important;
    opacity: 1 !important;
    margin: 1.2rem 0 !important;
}}

.answer-card-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    color: {t['gold']} !important;
    text-transform: uppercase;
    margin-bottom: 8px;
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {t['surface']} !important;
    border: 1.5px solid {t['line']} !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
}}

.stCaption, [data-testid="stCaptionContainer"] {{
    color: {t['muted']} !important;
}}

.stButton>button, [data-testid="stButton"] button {{
    background-color: {t['surface']};
    color: {t['ink']} !important;
    border: 1px solid {t['line']};
    font-weight: 500;
}}

.stButton>button p, [data-testid="stButton"] button p,
.stButton>button div, [data-testid="stButton"] button div {{
    color: {t['ink']} !important;
}}

.snapshot-card {{
    background: {t['surface']};
    border: 1px solid {t['line']};
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 16px;
}}

.sector-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    color: white !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    margin-bottom: 8px;
}}

.stat-row {{
    font-family: 'IBM Plex Mono', monospace;
    color: {t['muted']};
    font-size: 0.85rem;
    margin-top: 6px;
}}

.bookmark-tab {{
    display: inline-flex;
    align-items: center;
    background: {t['ink']};
    color: {t['paper']} !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 4px 10px 4px 12px;
    margin: 4px 6px 4px 0;
    border-radius: 0 4px 4px 0;
    position: relative;
}}

.bookmark-tab::before {{
    content: "";
    position: absolute;
    left: -8px;
    top: 0;
    border-top: 12px solid {t['ink']};
    border-bottom: 12px solid {t['ink']};
    border-left: 8px solid transparent;
}}

.source-note {{
    font-family: 'IBM Plex Mono', monospace;
    color: {t['gold']} !important;
    font-size: 0.85rem;
    margin-top: 10px;
}}

.chunk-block {{
    background: {t['surface']};
    border-left: 3px solid {t['gold']};
    padding: 12px 16px;
    margin-bottom: 12px;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
    color: {t['ink']} !important;
}}

/* Responsive adjustments for tablets and phones */
@media (max-width: 768px) {{
    .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1.5rem !important;
    }}

    h1, h2, h3, h4 {{
        font-size: 1.15rem !important;
    }}

    .snapshot-card {{
        padding: 12px;
    }}

    .stat-row {{
        font-size: 0.8rem;
    }}

    .sector-badge {{
        font-size: 0.7rem;
        padding: 3px 8px;
    }}

    .bookmark-tab {{
        font-size: 0.75rem;
        padding: 3px 8px 3px 10px;
    }}

    .chunk-block {{
        padding: 10px 12px;
        font-size: 0.85rem;
    }}

    .source-note {{
        font-size: 0.78rem;
    }}
}}

@media (max-width: 480px) {{
    h1, h2, h3, h4 {{
        font-size: 1.05rem !important;
    }}

    .snapshot-card {{
        padding: 10px;
    }}

    .stButton>button {{
        font-size: 0.85rem;
        padding: 0.4rem 0.6rem;
    }}
}}

/* Prevent horizontal scroll on any screen size */
.stApp, .block-container {{
    overflow-x: hidden;
    max-width: 100%;
}}

img, .snapshot-card, .chunk-block {{
    max-width: 100%;
}}</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Top header: title, theme toggle, company selector
# ---------------------------------------------------------------
header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown("### 📑 Company Report Assistant")
    st.caption("Ask questions grounded in real annual reports.")

with header_right:
    current_dark = st.session_state.get("dark_mode", False)
    toggle_label = "🌙 Dark mode" if current_dark else "☀️ Light mode"
    st.toggle(toggle_label, key="dark_mode")

st.markdown("**Choose a company**")
company = st.radio(
    "Choose a company",
    list(COMPANIES.keys()),
    horizontal=True,
    label_visibility="collapsed",
)

meta = COMPANY_META[company]
sector_name = COMPANIES[company]["sector"]
color = SECTOR_COLORS[meta["sector_key"]]

st.markdown(f"""
<div class="snapshot-card">
    <span class="sector-badge" style="background:{color}">{sector_name}</span>
    <div class="stat-row">{meta['pages']} pages in report</div>
</div>
""", unsafe_allow_html=True)

st.markdown("**Try asking:**")
chip_cols = st.columns(len(SUGGESTED_QUESTIONS[company]))
for col, q in zip(chip_cols, SUGGESTED_QUESTIONS[company]):
    with col:
        if st.button(q, key=f"suggest_{q}", use_container_width=True):
            st.session_state["question_input"] = q

st.markdown("---")


# ---------------------------------------------------------------
# Main area: Ask / Sources tabs
# ---------------------------------------------------------------
ask_tab, sources_tab = st.tabs(["Ask", "Sources"])

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

with ask_tab:
    st.markdown(f"#### Ask about {company}'s annual report")

    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="e.g. What did the company say about supply chain risk?",
    )

    just_answered = False
    loading_placeholder = st.empty()

    if st.button("🔍 Ask", use_container_width=False):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            loading_placeholder.markdown("⏳ **Generating your answer, please wait...**")

            with st.spinner(f"Reading {company}'s report and preparing an answer..."):
                vector_store = get_vector_store()
                llm = get_llm()
                chunks = retrieve_chunks(vector_store, company, question)

                if chunks:
                    context = build_context_string(chunks)
                    prompt = PROMPT_TEMPLATE.format(company=company, context=context, question=question)

            if not chunks:
                loading_placeholder.empty()
                st.session_state["last_result"] = {
                    "answer": "No relevant content was found in this company's report for that question.",
                    "sources": [],
                }
            else:
                st.markdown("---")
                st.markdown('<div class="answer-card-label">Answer</div>', unsafe_allow_html=True)

                def _stream_text():
                    first_chunk = True
                    for part in llm.stream(prompt):
                        if first_chunk:
                            loading_placeholder.empty()
                            first_chunk = False
                        yield part.content

                full_answer = st.write_stream(_stream_text())

                pages = sorted(set(s.metadata.get("page", 0) for s in chunks))
                st.markdown(
                    f'<div class="source-note">Grounded in {len(chunks)} '
                    f'excerpt(s) from {len(pages)} page(s) of the report. '
                    f'See the Sources tab for full detail.</div>',
                    unsafe_allow_html=True,
                )

                st.session_state["last_result"] = {"answer": full_answer, "sources": chunks}
                just_answered = True

    result = st.session_state["last_result"]
    if result and not just_answered:
        st.markdown("---")
        st.markdown('<div class="answer-card-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(result["answer"])

        if result["sources"]:
            pages = sorted(set(s.metadata.get("page", 0) for s in result["sources"]))
            st.markdown(
                f'<div class="source-note">Grounded in {len(result["sources"])} '
                f'excerpt(s) from {len(pages)} page(s) of the report. '
                f'See the Sources tab for full detail.</div>',
                unsafe_allow_html=True,
            )

with sources_tab:
    result = st.session_state["last_result"]
    if not result or not result["sources"]:
        st.caption("Ask a question first to see the exact report excerpts used.")
    else:
        st.markdown(f"#### Excerpts used for the last answer")
        for source in result["sources"]:
            page = source.metadata.get("page", "unknown")
            st.markdown(f'<span class="bookmark-tab">p. {page}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="chunk-block">{source.page_content}</div>', unsafe_allow_html=True)