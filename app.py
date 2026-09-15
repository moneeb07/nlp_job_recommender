"""NLP Job Recommender — Streamlit app.

Sessions 1-2: resume text extraction, the spaCy preprocessing pipeline, and the
parsed resume profile (NER + gazetteer skill matching).
"""

import html
import io

import pandas as pd
import spacy
import streamlit as st

from src.config import MAX_TOKEN_ROWS
from src.pdf_parser import extract_text_from_pdf
from src.preprocessing import get_token_analysis, preprocess_text
from src.resume_parser import ResumeParser, get_profile_text

st.set_page_config(page_title="NLP Job Recommender", layout="wide")


@st.cache_resource(show_spinner=False)
def get_parser() -> ResumeParser:
    return ResumeParser()


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


@st.cache_data(show_spinner=False)
def load_profile(raw_text: str) -> dict:
    return get_parser().parse(raw_text)


def render_chips(values: list[str]) -> None:
    if not values:
        st.caption("None found.")
        return

    chips = "".join(
        "<span style='display:inline-block;padding:3px 10px;margin:0 6px 6px 0;"
        "border:1px solid #4A4A46;border-radius:12px;background:#30302E;"
        f"font-size:0.85rem;'>{html.escape(value)}</span>"
        for value in values
    )
    st.markdown(chips, unsafe_allow_html=True)


st.title("NLP Job Recommender")
st.caption(
    "Sessions 1-2 — PDF extraction, the spaCy preprocessing pipeline, and resume "
    "parsing with named entity recognition and gazetteer skill matching."
)

uploaded = st.file_uploader("Resume (PDF)", type="pdf")

if uploaded is None:
    st.info("Upload a PDF resume to inspect each stage of the pipeline.")
    st.stop()

with st.spinner("Extracting and parsing…"):
    raw_text = load_raw_text(uploaded.getvalue())

if not raw_text.strip():
    st.error(
        "No text could be extracted. This PDF is likely a scanned image, "
        "which needs OCR rather than text extraction."
    )
    st.stop()

cleaned_text = load_cleaned_text(raw_text)
tokens = load_token_analysis(raw_text)
profile = load_profile(raw_text)

left, middle, right = st.columns(3)
left.metric("Characters extracted", f"{len(raw_text):,}")
middle.metric("Skills matched", len(profile["skills"]))
right.metric("Years of experience", profile["years_of_experience"])

profile_tab, raw_tab, cleaned_tab, token_tab = st.tabs(
    ["Profile", "Raw text", "Preprocessed text", "Token analysis"]
)

with profile_tab:
    summary, contact = st.columns([2, 1])

    with summary:
        st.subheader("Skills")
        st.caption(
            f"{len(profile['skills'])} matched against the "
            "skills taxonomy using spaCy's PhraseMatcher."
        )
        render_chips(profile["skills"])

        st.subheader("Education")
        if profile["education"]:
            for line in profile["education"]:
                st.markdown(f"- {line}")
        else:
            st.caption("No degree statement found.")

        st.subheader("Job titles")
        render_chips(profile["job_titles"])

    with contact:
        st.subheader("Details")
        for label, key in [
            ("Name", "name"),
            ("Email", "email"),
            ("Phone", "phone"),
        ]:
            st.text_input(label, profile[key] or "—", disabled=True)

        st.markdown("**Organizations**")
        render_chips(profile["organizations"])

        st.markdown("**Locations**")
        render_chips(profile["locations"])

    with st.expander(f"spaCy named entities ({len(profile['entities'])} found)"):
        st.caption(
            "spaCy's NER model was trained on OntoNotes, which has no SKILL "
            "label — that is why skills are matched separately by gazetteer."
        )
        entity_rows = [
            {
                "text": text,
                "label": label,
                "what the label means": spacy.explain(label) or "—",
            }
            for text, label in profile["entities"]
        ]
        if entity_rows:
            st.dataframe(
                pd.DataFrame(entity_rows),
                width="stretch",
                hide_index=True,
                height=320,
            )
        else:
            st.caption("No entities found.")

    with st.expander("Profile text used for embedding (session 5 input)"):
        st.code(get_profile_text(profile) or "—", language=None, wrap_lines=True)

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
