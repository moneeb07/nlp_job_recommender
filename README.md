# NLP Job Recommender

A semantic job recommendation system built on classical and neural NLP — resume
parsing with spaCy, sparse retrieval with TF-IDF, dense retrieval with
Sentence-BERT and FAISS, and a two-stage retrieve-and-rank pipeline. No LLM.

**Status:** Session 1 of 7 — environment, project skeleton, PDF extraction, and
the spaCy preprocessing pipeline.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Python 3.11 or 3.12 is recommended; several of these libraries do not yet ship
wheels for 3.13+.

## Run

```bash
source venv/bin/activate
streamlit run app.py
```

Then upload a PDF resume to see the extracted text, the preprocessed text, and a
token-level breakdown of the spaCy pipeline.

## Tests

```bash
pytest
```

## Layout

```
data/raw/          job postings dataset (gitignored)
data/processed/    cleaned job data
models/            FAISS index and fitted vectorizers (gitignored)
src/config.py      all paths and tunable constants
src/pdf_parser.py  PDF text extraction
src/preprocessing.py  tokenization, lemmatization, stopword removal
app.py             Streamlit UI
```

## NLP techniques

| Concept | Where |
|---|---|
| Text extraction from unstructured documents | `src/pdf_parser.py` (pdfplumber) |
| Tokenization | `src/preprocessing.py` (spaCy) |
| Lemmatization | `src/preprocessing.py` (spaCy) |
| Stopword and punctuation removal | `src/preprocessing.py` (spaCy) |
| Part-of-speech tagging | `src/preprocessing.py` (spaCy) |

Later sessions add NER, a skills gazetteer, TF-IDF, Sentence-BERT embeddings,
FAISS retrieval, and IR evaluation metrics.
