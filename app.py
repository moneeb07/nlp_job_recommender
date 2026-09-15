"""NLP Job Recommender — Streamlit app.

Sessions 1-5: PDF extraction, the spaCy preprocessing pipeline, resume parsing,
the TF-IDF baseline, and semantic retrieval with Sentence-BERT + FAISS.
"""

import html
import io

import pandas as pd
import spacy
import streamlit as st

from src.baseline_tfidf import TFIDFRecommender
from src.config import JOBS_PARQUET, MAX_TOKEN_ROWS, TOP_K
from src.pdf_parser import extract_text_from_pdf
from src.preprocessing import get_token_analysis, preprocess_text
from src.recommender import JobRecommender
from src.resume_parser import ResumeParser, get_profile_text

st.set_page_config(page_title="NLP Job Recommender", layout="wide")

COMPONENT_COLOURS = {
    "semantic": "#D97757",
    "skills": "#7A9E8E",
    "experience": "#6B7FA3",
}


@st.cache_resource(show_spinner=False)
def get_parser() -> ResumeParser:
    return ResumeParser()


@st.cache_resource(show_spinner="Loading job corpus…")
def get_jobs() -> pd.DataFrame:
    return pd.read_parquet(JOBS_PARQUET)


@st.cache_resource(show_spinner="Loading TF-IDF baseline…")
def get_tfidf() -> TFIDFRecommender:
    return TFIDFRecommender.load()


@st.cache_resource(show_spinner="Loading Sentence-BERT index…")
def get_semantic() -> JobRecommender:
    return JobRecommender.load()


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


def render_chips(values: list[str], colour: str | None = None) -> None:
    if not values:
        st.caption("None found.")
        return

    border = colour or "#4A4A46"
    chips = "".join(
        "<span style='display:inline-block;padding:3px 10px;margin:0 6px 6px 0;"
        f"border:1px solid {border};border-radius:12px;background:#30302E;"
        f"font-size:0.85rem;'>{html.escape(value)}</span>"
        for value in values
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_score_breakdown(result: dict) -> None:
    """Stacked bar showing how the three weighted components make the score.

    Each segment's width is that component's contribution to the final score,
    so the bar's total length is the score itself.
    """
    recommender = get_semantic()
    segments = [
        ("semantic", result["semantic_score"] * recommender.weight_semantic),
        ("skills", result["skill_overlap"] * recommender.weight_skills),
        ("experience", result["experience_match"] * recommender.weight_experience),
    ]

    bars = "".join(
        f"<div title='{name}: {value:.3f}' style='width:{value * 100:.2f}%;"
        f"background:{COMPONENT_COLOURS[name]};height:10px;'></div>"
        for name, value in segments
    )
    st.markdown(
        f"<div style='display:flex;width:100%;background:#30302E;border-radius:5px;"
        f"overflow:hidden;margin:6px 0;'>{bars}</div>",
        unsafe_allow_html=True,
    )

    legend = " &nbsp; ".join(
        f"<span style='color:{COMPONENT_COLOURS[name]};'>&#9632;</span> "
        f"<span style='font-size:0.78rem;color:#A8A79F;'>{name} {value:.3f}</span>"
        for name, value in segments
    )
    st.markdown(legend, unsafe_allow_html=True)


def render_job_card(job: pd.Series, score: float, body=None) -> None:
    with st.container(border=True):
        header, score_column = st.columns([5, 1])
        header.markdown(f"**{html.escape(str(job['title']))}**")
        header.caption(f"{job['company']} · {job['location'] or 'Location not stated'}")
        score_column.metric("score", f"{score:.3f}", label_visibility="collapsed")
        if body is not None:
            body()


st.title("NLP Job Recommender")
st.caption(
    "Sessions 1-5 — resume parsing with spaCy, a TF-IDF baseline, and "
    "semantic retrieval with Sentence-BERT and FAISS."
)

tfidf_ready = TFIDFRecommender.is_available()
semantic_ready = JobRecommender.is_available() and JOBS_PARQUET.exists()

with st.sidebar:
    st.subheader("Retrieval")
    mode = st.radio(
        "Mode",
        ["Semantic (Sentence-BERT)", "Baseline (TF-IDF)"],
        help="The baseline matches on shared words; the semantic mode matches on meaning.",
    )
    top_k = st.slider("Results", min_value=3, max_value=20, value=TOP_K)

    st.divider()
    st.caption("Model status")
    st.caption(f"{'ready' if tfidf_ready else 'not built'} — TF-IDF baseline")
    st.caption(f"{'ready' if semantic_ready else 'not built'} — FAISS index")

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

# The two systems need the profile in different forms. TF-IDF compares against a
# lemmatized corpus, so the query is lemmatized the same way. Sentence-BERT was
# trained on ordinary prose and the index was built from raw job text, so it
# gets the profile unprocessed.
profile_text = get_profile_text(profile)
tfidf_query = preprocess_text(profile_text)

left, middle, right = st.columns(3)
left.metric("Characters extracted", f"{len(raw_text):,}")
middle.metric("Skills matched", len(profile["skills"]))
right.metric("Years of experience", profile["years_of_experience"])

recommend_tab, compare_tab, profile_tab, raw_tab, cleaned_tab, token_tab = st.tabs(
    ["Recommend", "Compare", "Profile", "Raw text", "Preprocessed text", "Token analysis"]
)


def tfidf_results(k: int) -> list[dict]:
    return get_tfidf().recommend(tfidf_query, top_k=k)


def semantic_results(k: int) -> list[dict]:
    return get_semantic().recommend(profile, top_k=k)


with recommend_tab:
    using_semantic = mode.startswith("Semantic")

    if using_semantic and not semantic_ready:
        st.warning("The FAISS index has not been built. Run `python -m src.build_index`.")
    elif not using_semantic and not tfidf_ready:
        st.warning("The baseline has not been fitted. Run `python -m src.baseline_tfidf`.")
    elif not profile_text.strip():
        st.warning("No skills, job titles or education were found to match on.")
    else:
        jobs = get_jobs()

        if using_semantic:
            st.caption(
                "Two stages: FAISS retrieves the 100 nearest postings, then they are "
                "re-ranked on semantic similarity, skill overlap and experience fit."
            )
            for result in semantic_results(top_k):
                job = jobs.iloc[result["index"]]

                def body(result=result):
                    render_score_breakdown(result)
                    if result["matched_skills"]:
                        st.caption("Skills you have")
                        render_chips(result["matched_skills"], COMPONENT_COLOURS["skills"])
                    if result["missing_skills"]:
                        st.caption("Skills the posting asks for that you did not list")
                        render_chips(result["missing_skills"][:12], "#7A6A55")

                render_job_card(job, result["score"], body)
        else:
            st.caption(
                "Cosine similarity between TF-IDF vectors. Terms below are those "
                "contributing most to each match."
            )
            model = get_tfidf()
            for result in tfidf_results(top_k):
                job = jobs.iloc[result["index"]]

                def body(result=result, model=model):
                    terms = [term for term, _ in model.shared_terms(tfidf_query, result["index"])]
                    st.caption("Top contributing terms")
                    render_chips(terms or ["no shared terms"])

                render_job_card(job, result["score"], body)

with compare_tab:
    if not (tfidf_ready and semantic_ready):
        st.warning("Both systems must be built to compare them.")
    else:
        st.caption(
            "The same resume through both systems. TF-IDF can only match shared "
            "surface forms; the bi-encoder matches meaning, so it finds postings "
            "that use different vocabulary for the same work."
        )
        jobs = get_jobs()
        sparse = tfidf_results(10)
        dense = semantic_results(10)
        sparse_column, dense_column = st.columns(2)

        for column, heading, results in [
            (sparse_column, "TF-IDF", sparse),
            (dense_column, "Sentence-BERT", dense),
        ]:
            with column:
                st.subheader(heading)
                for rank, result in enumerate(results, start=1):
                    job = jobs.iloc[result["index"]]
                    st.markdown(f"{rank}. **{job['title']}** — {job['company']}")
                    st.caption(f"score {result['score']:.3f}")

        overlap = {r["index"] for r in sparse} & {r["index"] for r in dense}
        st.caption(f"{len(overlap)} of 10 postings appear in both top tens.")

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
        for label, key in [("Name", "name"), ("Email", "email"), ("Phone", "phone")]:
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
            {"text": text, "label": label, "what the label means": spacy.explain(label) or "—"}
            for text, label in profile["entities"]
        ]
        if entity_rows:
            st.dataframe(pd.DataFrame(entity_rows), width="stretch", hide_index=True, height=320)
        else:
            st.caption("No entities found.")

    with st.expander("Profile text used for retrieval"):
        st.code(profile_text or "—", language=None, wrap_lines=True)

with raw_tab:
    st.text_area("Extracted text", raw_text, height=520, label_visibility="collapsed")

with cleaned_tab:
    st.text_area("Preprocessed text", cleaned_text, height=520, label_visibility="collapsed")

with token_tab:
    hide_stopwords = st.checkbox("Hide stopwords and punctuation", value=False)

    table = tokens
    if hide_stopwords:
        table = table[~table["is_stopword"] & (table["POS"] != "PUNCT")]

    if len(table) > MAX_TOKEN_ROWS:
        st.caption(f"Showing the first {MAX_TOKEN_ROWS:,} of {len(table):,} tokens.")
        table = table.head(MAX_TOKEN_ROWS)

    st.dataframe(table, width="stretch", height=520, hide_index=True)
