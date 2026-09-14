"""NLP Job Recommender — Streamlit app.

Session 1: resume text extraction and the spaCy preprocessing pipeline.
"""

import io

import pandas as pd
import streamlit as st

from src.config import MAX_TOKEN_ROWS
from src.pdf_parser import extract_text_from_pdf
from src.preprocessing import get_token_analysis, preprocess_text

st.set_page_config(page_title="NLP Job Recommender", layout="wide")


@st.cache_data(show_spinner=False)
def load_raw_text(pdf_bytes: bytes) -> str:
    return extract_text_from_pdf(io.BytesIO(pdf_bytes))


@st.cache_data(show_spinner=False)
def load_cleaned_text(raw_text: str) -> str:
    return preprocess_text(raw_text)


@st.cache_data(show_spinner=False)
def load_token_analysis(raw_text: str) -> pd.DataFrame:
    rows = get_token_analysis(raw_text)
    return pd.DataFrame(rows, columns=["token", "lemma", "POS", "is_stopword"])


st.title("NLP Job Recommender")
st.caption(
    "Session 1 — PDF text extraction and the spaCy preprocessing pipeline "
    "(tokenization, lemmatization, stopword and punctuation removal)."
)

uploaded = st.file_uploader("Resume (PDF)", type="pdf")

if uploaded is None:
    st.info("Upload a PDF resume to inspect each stage of the pipeline.")
    st.stop()

with st.spinner("Extracting and preprocessing…"):
    raw_text = load_raw_text(uploaded.getvalue())

if not raw_text.strip():
    st.error(
        "No text could be extracted. This PDF is likely a scanned image, "
        "which needs OCR rather than text extraction."
    )
    st.stop()

cleaned_text = load_cleaned_text(raw_text)
tokens = load_token_analysis(raw_text)

left, middle, right = st.columns(3)
left.metric("Characters extracted", f"{len(raw_text):,}")
middle.metric("Tokens", f"{len(tokens):,}")
right.metric("Tokens after cleaning", f"{len(cleaned_text.split()):,}")

raw_tab, cleaned_tab, token_tab = st.tabs(
    ["Raw text", "Preprocessed text", "Token analysis"]
)

with raw_tab:
    st.text_area("Extracted text", raw_text, height=520, label_visibility="collapsed")

with cleaned_tab:
    st.text_area(
        "Preprocessed text", cleaned_text, height=520, label_visibility="collapsed"
    )

with token_tab:
    hide_stopwords = st.checkbox("Hide stopwords and punctuation", value=False)

    table = tokens
    if hide_stopwords:
        table = table[~table["is_stopword"] & (table["POS"] != "PUNCT")]

    if len(table) > MAX_TOKEN_ROWS:
        st.caption(f"Showing the first {MAX_TOKEN_ROWS:,} of {len(table):,} tokens.")
        table = table.head(MAX_TOKEN_ROWS)

    st.dataframe(table, width="stretch", height=520, hide_index=True)
