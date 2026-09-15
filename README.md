# NLP Job Recommender

A semantic job recommendation system built on classical and neural NLP — resume
parsing with spaCy, sparse retrieval with TF-IDF, dense retrieval with
Sentence-BERT and FAISS, and a two-stage retrieve-and-rank pipeline. No LLM.

**Status:** Sessions 1-3 of 7 — environment and skeleton, resume parsing with
NER, and the job postings dataset pipeline.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 or 3.12 is recommended; several of these libraries do not yet ship
wheels for 3.13+. The spaCy model is pinned in `requirements.txt`, so no
separate `spacy download` step is needed.

## Build the job corpus

Put the LinkedIn Job Postings CSV at `data/raw/postings.csv`, then:

```bash
python -m src.data_loader
```

This samples 5,000 postings (configurable in `src/config.py`), extracts skills
and experience requirements, and writes `data/processed/jobs.parquet`. It takes
about two minutes.

## Run

```bash
source venv/bin/activate
streamlit run app.py
```

Upload a PDF resume to see the parsed profile, the extracted text, the
preprocessed text, and a token-level breakdown of the spaCy pipeline.

## Tests

```bash
pytest
```

## Layout

```
data/raw/               job postings dataset (gitignored)
data/processed/         jobs.parquet, built by src/data_loader.py
data/skills_taxonomy.txt  ~440 skills, the shared gazetteer (tracked)
models/                 FAISS index and fitted vectorizers (gitignored)
src/config.py           all paths and tunable constants
src/pdf_parser.py       PDF text extraction
src/preprocessing.py    tokenization, lemmatization, stopword removal
src/resume_parser.py    NER, skill matching, and profile fields
src/data_loader.py      job corpus loading, cleaning, and enrichment
notebooks/              exploratory analysis
app.py                  Streamlit UI
```

## NLP techniques

| Concept | Where |
|---|---|
| Text extraction from unstructured documents | `src/pdf_parser.py` (pdfplumber) |
| Tokenization | `src/preprocessing.py` (spaCy) |
| Lemmatization | `src/preprocessing.py` (spaCy) |
| Stopword and punctuation removal | `src/preprocessing.py` (spaCy) |
| Part-of-speech tagging | `src/preprocessing.py` (spaCy) |
| Named entity recognition | `src/resume_parser.py` (spaCy NER) |
| Gazetteer / pattern matching | `src/resume_parser.py` (spaCy `PhraseMatcher`) |

Later sessions add TF-IDF, Sentence-BERT embeddings, FAISS retrieval, hybrid
re-ranking, and IR evaluation metrics.

## Notes on the data

Resumes and job postings are matched against **one** skills vocabulary
(`data/skills_taxonomy.txt`) using the same `PhraseMatcher`, so the skill overlap
score in session 5 compares like with like.

Two taxonomy decisions were made by measuring against the corpus rather than by
intuition:

- Bare `c`, `r` and `go` were removed. They matched `b/c`, `Class C licence`,
  `R&D` and "the go-to person" far more often than the languages, so the
  qualified forms (`c programming`, `golang`) are used instead.
- Skills are extracted from the **full** description rather than the truncated
  embedding text. Requirement lists usually sit past the 1,500-character cutoff;
  truncating left 47% of postings with no skills at all, versus 19% in full.

The remaining 19% are genuinely non-technical postings, which a technical
taxonomy is expected to miss.
