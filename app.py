"""NLP Job Recommender — Streamlit app.

Resume in, ranked job postings out, with every stage of the NLP pipeline and the
retrieval benchmark visible alongside the results.
"""

import html
import io
import json

import pandas as pd
import plotly.express as px
import spacy
import streamlit as st

from src.baseline_tfidf import TFIDFRecommender
from src.config import (
    JOBS_PARQUET,
    MAX_TOKEN_ROWS,
    RESULTS_DIR,
    TOP_K,
    WEIGHT_EXPERIENCE,
    WEIGHT_SEMANTIC,
    WEIGHT_SKILL_OVERLAP,
)
from src.explainer_rules import explain
from src.pdf_parser import extract_text_from_pdf
from src.preprocessing import get_token_analysis, preprocess_text
from src.recommender import JobRecommender
from src.resume_parser import ResumeParser, get_profile_text

st.set_page_config(page_title="NLP Job Recommender", layout="wide")

TFIDF_MODE = "Baseline (TF-IDF)"
SEMANTIC_MODE = "Semantic (Sentence-BERT)"
FULL_MODE = "Full pipeline (semantic + re-ranking)"

COMPONENT_COLOURS = {
    "semantic": "#D97757",
    "skills": "#7A9E8E",
    "experience": "#6B7FA3",
}
MATCHED_COLOUR = "#7A9E8E"
MISSING_COLOUR = "#B8823C"

READINESS_COLOURS = {
    "strong fit": "#7A9E8E",
    "good fit": "#6B7FA3",
    "stretch role": "#B8823C",
}


# --------------------------------------------------------------------------
# Cached loaders — models are loaded once per session, not per interaction
# --------------------------------------------------------------------------

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
def load_results(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return path.read_text()


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


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------

def render_chips(values, colour: str | None = None) -> None:
    values = list(values or [])
    if not values:
        st.caption("None found.")
        return

    border = colour or "#4A4A46"
    chips = "".join(
        "<span style='display:inline-block;padding:3px 10px;margin:0 6px 6px 0;"
        f"border:1px solid {border};border-radius:12px;background:#30302E;"
        f"font-size:0.85rem;'>{html.escape(str(value))}</span>"
        for value in values
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_score_bar(score: float, colour: str = "#D97757") -> None:
    width = max(0.0, min(score, 1.0)) * 100
    st.markdown(
        "<div style='width:100%;background:#30302E;border-radius:5px;overflow:hidden;"
        f"margin:4px 0;'><div style='width:{width:.1f}%;background:{colour};"
        "height:10px;'></div></div>",
        unsafe_allow_html=True,
    )


def render_score_breakdown(result: dict, weights: tuple[float, float, float]) -> None:
    """Stacked bar whose segments are each component's contribution.

    The bar's total length is the final score, so the decomposition is exact.
    """
    semantic_weight, skills_weight, experience_weight = weights
    segments = [
        ("semantic", result["semantic_score"] * semantic_weight),
        ("skills", result["skill_overlap"] * skills_weight),
        ("experience", result["experience_match"] * experience_weight),
    ]

    bars = "".join(
        f"<div title='{name}: {value:.3f}' style='width:{value * 100:.2f}%;"
        f"background:{COMPONENT_COLOURS[name]};height:10px;'></div>"
        for name, value in segments
    )
    st.markdown(
        "<div style='display:flex;width:100%;background:#30302E;border-radius:5px;"
        f"overflow:hidden;margin:6px 0;'>{bars}</div>",
        unsafe_allow_html=True,
    )

    legend = " &nbsp; ".join(
        f"<span style='color:{COMPONENT_COLOURS[name]};'>&#9632;</span> "
        f"<span style='font-size:0.78rem;color:#A8A79F;'>{name} {value:.3f}</span>"
        for name, value in segments
    )
    st.markdown(legend, unsafe_allow_html=True)


def render_readiness(label: str) -> None:
    colour = READINESS_COLOURS.get(label, "#4A4A46")
    st.markdown(
        f"<span style='display:inline-block;padding:2px 10px;border-radius:12px;"
        f"border:1px solid {colour};color:{colour};font-size:0.78rem;"
        f"text-transform:uppercase;letter-spacing:0.04em;'>{html.escape(label)}</span>",
        unsafe_allow_html=True,
    )


def render_job_card(job: pd.Series, score: float, body=None) -> None:
    with st.container(border=True):
        header, score_column = st.columns([5, 1])
        header.markdown(f"**{html.escape(str(job['title']))}**")
        header.caption(f"{job['company']} · {job['location'] or 'Location not stated'}")
        score_column.metric("score", f"{score:.3f}", label_visibility="collapsed")
        render_score_bar(score)
        if body is not None:
            body()


# --------------------------------------------------------------------------
# Header and sidebar
# --------------------------------------------------------------------------

st.title("NLP Job Recommender")
st.caption(
    "Matches a resume against 5,000 job postings using spaCy for parsing, "
    "Sentence-BERT and FAISS for retrieval, and a hybrid re-ranker for the final order."
)

tfidf_ready = TFIDFRecommender.is_available()
semantic_ready = JobRecommender.is_available() and JOBS_PARQUET.exists()

with st.sidebar:
    st.subheader("Retrieval")
    mode = st.radio("Mode", [FULL_MODE, SEMANTIC_MODE, TFIDF_MODE])
    top_k = st.slider("Results", min_value=3, max_value=20, value=TOP_K)

    st.subheader("Re-ranking weights")
    st.caption("Applies to the full pipeline. Session 6 ablates these.")
    semantic_weight = st.slider("Semantic similarity", 0.0, 1.0, WEIGHT_SEMANTIC, 0.05)
    skills_weight = st.slider("Skill overlap", 0.0, 1.0, WEIGHT_SKILL_OVERLAP, 0.05)
    experience_weight = st.slider("Experience match", 0.0, 1.0, WEIGHT_EXPERIENCE, 0.05)

    total = semantic_weight + skills_weight + experience_weight
    if total == 0:
        st.warning("All weights are zero — nothing to rank on.")
    elif abs(total - 1.0) > 0.001:
        st.caption(f"Weights sum to {total:.2f}; scores are not comparable to the default.")

    st.divider()
    st.caption("Model status")
    st.caption(f"{'ready' if tfidf_ready else 'not built'} — TF-IDF baseline")
    st.caption(f"{'ready' if semantic_ready else 'not built'} — FAISS index")

uploaded = st.file_uploader("Resume (PDF)", type="pdf")

recommend_tab, pipeline_tab, evaluation_tab, compare_tab = st.tabs(
    ["Recommend", "NLP Pipeline", "Evaluation", "Compare"]
)


# --------------------------------------------------------------------------
# Evaluation tab does not need a resume, so it renders either way
# --------------------------------------------------------------------------

def render_evaluation() -> None:
    benchmark = load_results("benchmark.json")
    if benchmark is None:
        st.warning(
            "No benchmark yet. Run `python -m src.build_eval_set` then "
            "`python -m src.evaluate`."
        )
        return

    st.caption(
        "Labels are heuristic, not human judgements — a posting is relevant when "
        "it shares a normalized title family with the query and its skills overlap "
        "by more than 0.4 Jaccard. The numbers compare systems on equal terms; "
        "their absolute level means little."
    )

    frame = pd.DataFrame(benchmark).T
    ndcg_columns = sorted(
        (column for column in frame.columns if column.startswith("NDCG")),
        key=lambda name: int(name.split("@")[1]),
    )
    st.subheader("Benchmark")
    st.dataframe(frame.round(3), width="stretch")

    # Grouped, not stacked: these are three independent measurements of the
    # same system, and stacking them implies a total that means nothing.
    chart_data = (
        frame[ndcg_columns]
        .reset_index(names="system")
        .melt(id_vars="system", var_name="cutoff", value_name="NDCG")
    )
    figure = px.bar(
        chart_data,
        x="cutoff",
        y="NDCG",
        color="system",
        barmode="group",
        template="plotly_dark",
        height=360,
        color_discrete_sequence=[
            COMPONENT_COLOURS["semantic"],
            COMPONENT_COLOURS["skills"],
            COMPONENT_COLOURS["experience"],
        ],
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.2, "title": ""},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title="",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(figure, width="stretch")

    ablation = load_results("ablation.json")
    if ablation:
        st.subheader("Re-ranking weight ablation")
        st.caption(
            "Standard errors overlap across the middle of this table: on 50 "
            "heuristic queries, a few points of NDCG is not evidence."
        )
        ablation_frame = pd.DataFrame(ablation).T
        ablation_frame["weights"] = ablation_frame["weights"].map(
            lambda row: " / ".join(f"{value:.2f}" for value in row)
        )
        st.dataframe(
            ablation_frame[["weights", "NDCG@10", "stderr", "P@10", "MAP", "wins", "losses"]]
            .sort_values("NDCG@10", ascending=False)
            .round(3),
            width="stretch",
        )


with evaluation_tab:
    render_evaluation()


if uploaded is None:
    with recommend_tab:
        st.info("Upload a PDF resume to get recommendations.")
    with pipeline_tab:
        st.info("Upload a PDF resume to inspect the NLP pipeline.")
    with compare_tab:
        st.info("Upload a PDF resume to compare the two retrieval systems.")
    st.stop()


# --------------------------------------------------------------------------
# Everything below needs a resume
# --------------------------------------------------------------------------

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
weights = (semantic_weight, skills_weight, experience_weight)


def tfidf_results(k: int) -> list[dict]:
    return get_tfidf().recommend(tfidf_query, top_k=k)


def semantic_results(k: int, rerank: bool = True) -> list[dict]:
    recommender = get_semantic()
    if not rerank:
        return recommender.retrieve_only(profile, top_k=k)
    return recommender.with_weights(*weights).recommend(profile, top_k=k)


with recommend_tab:
    if mode == TFIDF_MODE and not tfidf_ready:
        st.warning("The baseline has not been fitted. Run `python -m src.baseline_tfidf`.")
    elif mode != TFIDF_MODE and not semantic_ready:
        st.warning("The FAISS index has not been built. Run `python -m src.build_index`.")
    elif not profile_text.strip():
        st.warning("No skills, job titles or education were found to match on.")
    else:
        jobs = get_jobs()

        if mode == TFIDF_MODE:
            st.caption(
                "Cosine similarity between TF-IDF vectors. The terms shown are "
                "those contributing most to each match."
            )
            model = get_tfidf()
            for result in tfidf_results(top_k):
                job = jobs.iloc[result["index"]]

                def body(result=result, model=model):
                    terms = [t for t, _ in model.shared_terms(tfidf_query, result["index"])]
                    st.caption("Top contributing terms")
                    render_chips(terms or ["no shared terms"])

                render_job_card(job, result["score"], body)

        elif mode == SEMANTIC_MODE:
            st.caption(
                "FAISS nearest neighbours by embedding similarity, with no "
                "re-ranking. This is the retrieval stage on its own."
            )
            for result in semantic_results(top_k, rerank=False):
                job = jobs.iloc[result["index"]]
                render_job_card(job, result["score"])

        else:
            st.caption(
                "Two stages: FAISS retrieves the 100 nearest postings, then they "
                "are re-ranked on semantic similarity, skill overlap and experience fit."
            )
            for result in semantic_results(top_k):
                job = jobs.iloc[result["index"]]
                explanation = explain(
                    result,
                    candidate_years=profile["years_of_experience"],
                    required_years=job["min_years_experience"],
                )

                def body(result=result, explanation=explanation):
                    render_readiness(explanation["readiness"])
                    st.write(explanation["text"])
                    render_score_breakdown(result, weights)
                    if result["matched_skills"]:
                        st.caption("Skills you have")
                        render_chips(result["matched_skills"], MATCHED_COLOUR)
                    if result["missing_skills"]:
                        st.caption("Skills the posting asks for that you did not list")
                        render_chips(result["missing_skills"][:12], MISSING_COLOUR)

                render_job_card(job, result["score"], body)


with pipeline_tab:
    st.caption(
        "Every stage between the uploaded PDF and the ranked list, in order."
    )

    summary, contact = st.columns([2, 1])
    with summary:
        st.subheader("Extracted skills")
        st.caption(
            f"{len(profile['skills'])} matched against the taxonomy with spaCy's "
            "PhraseMatcher, which compares token sequences rather than substrings."
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
        st.metric("Years of experience", profile["years_of_experience"])
        st.markdown("**Organizations**")
        render_chips(profile["organizations"])
        st.markdown("**Locations**")
        render_chips(profile["locations"])

    with st.expander(f"Named entities ({len(profile['entities'])} found)"):
        st.caption(
            "spaCy's NER model was trained on OntoNotes, which has no SKILL "
            "label — that is why skills are matched separately by gazetteer."
        )
        rows = [
            {"text": text, "label": label, "what the label means": spacy.explain(label) or "—"}
            for text, label in profile["entities"]
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=300)
        else:
            st.caption("No entities found.")

    with st.expander(f"Token analysis ({len(tokens):,} tokens)"):
        hide_stopwords = st.checkbox("Hide stopwords and punctuation", value=False)
        table = tokens
        if hide_stopwords:
            table = table[~table["is_stopword"] & (table["POS"] != "PUNCT")]
        if len(table) > MAX_TOKEN_ROWS:
            st.caption(f"Showing the first {MAX_TOKEN_ROWS:,} of {len(table):,} tokens.")
            table = table.head(MAX_TOKEN_ROWS)
        st.dataframe(table, width="stretch", height=320, hide_index=True)

    if tfidf_ready:
        with st.expander("Highest-weighted TF-IDF terms in this resume"):
            st.caption(
                "Term frequency times inverse document frequency: terms common "
                "across every posting are down-weighted towards zero."
            )
            terms = get_tfidf().get_top_terms(tfidf_query, n=25)
            st.dataframe(
                pd.DataFrame(terms, columns=["term", "weight"]),
                width="stretch", hide_index=True, height=300,
            )

    with st.expander("Raw extracted text"):
        st.text_area("Raw", raw_text, height=320, label_visibility="collapsed")

    with st.expander("Preprocessed text"):
        st.text_area("Cleaned", cleaned_text, height=320, label_visibility="collapsed")

    with st.expander("Profile text used for retrieval"):
        st.code(profile_text or "—", language=None, wrap_lines=True)


with compare_tab:
    if not (tfidf_ready and semantic_ready):
        st.warning("Both systems must be built to compare them.")
    elif not profile_text.strip():
        st.warning("No skills, job titles or education were found to match on.")
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
            (dense_column, "Sentence-BERT + re-ranking", dense),
        ]:
            with column:
                st.subheader(heading)
                for rank, result in enumerate(results, start=1):
                    job = jobs.iloc[result["index"]]
                    st.markdown(f"{rank}. **{job['title']}** — {job['company']}")
                    st.caption(f"score {result['score']:.3f}")

        overlap = {r["index"] for r in sparse} & {r["index"] for r in dense}
        st.caption(f"{len(overlap)} of 10 postings appear in both top tens.")
