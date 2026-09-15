# NLP Job Recommender

A semantic job recommendation system built on classical and neural NLP — resume
parsing with spaCy, sparse retrieval with TF-IDF, dense retrieval with
Sentence-BERT and FAISS, and a two-stage retrieve-and-rank pipeline. No LLM.

**Status:** Sessions 1-5 of 7 — environment and skeleton, resume parsing with
NER, the job postings dataset pipeline, the TF-IDF baseline, and semantic
retrieval with Sentence-BERT and FAISS.

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

Then fit the two retrieval systems:

```bash
python -m src.baseline_tfidf   # sparse baseline, ~10s
python -m src.build_index      # Sentence-BERT embeddings + FAISS index
```

The index build downloads the `all-MiniLM-L6-v2` weights (~80MB) on first run.
Both write to `models/`, which is gitignored — rebuild them after cloning.

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
src/baseline_tfidf.py   sparse TF-IDF baseline
src/embedder.py         Sentence-BERT wrapper, unit-normalized vectors
src/build_index.py      one-time FAISS index build
src/recommender.py      two-stage retrieve-and-rank
notebooks/              exploratory analysis
app.py                  Streamlit UI
```

## How retrieval works

**Baseline — TF-IDF (`src/baseline_tfidf.py`).** Each posting becomes a sparse
vector over a 50,000-term 1-2 gram vocabulary, weighted by term frequency times
inverse document frequency and compared by cosine similarity. It matches only on
shared surface forms: a resume saying "neural networks" and a posting saying
"deep learning" score exactly zero against each other. That limitation is the
reason the neural system exists, and both are kept so the two can be compared.

**Semantic — Sentence-BERT + FAISS (`src/recommender.py`).** A two-stage
retrieve-and-rank pipeline:

1. **Retrieval.** `all-MiniLM-L6-v2` embeds every posting once, offline, into a
   384-dimension vector. Vectors are L2-normalized, so FAISS `IndexFlatIP`
   inner-product search *is* cosine search. A query returns the 100 nearest
   postings.
2. **Re-ranking.** Those 100 are rescored on signals a single vector cannot
   carry:

   ```
   final = 0.50 * semantic_similarity
         + 0.30 * skill_overlap        (Jaccard over the shared taxonomy)
         + 0.20 * experience_match     (candidate years vs. stated requirement)
   ```

Scoring all 5,000 postings on every signal would waste effort on thousands that
are obviously irrelevant, while retrieval alone cannot tell that a candidate is
two years short of a requirement. The weights live in `src/config.py`; session 6
tests alternatives with an ablation study rather than taking them on faith.

### Measured on this corpus

| | TF-IDF | Sentence-BERT + re-rank |
|---|---|---|
| Model artefact | 50,000-term vocabulary, 0.31% dense | 5,000 × 384 index, 7.7 MB |
| Build time | ~10s | ~5.5 min (15 texts/s, CPU) |
| Query latency | ~49 ms | ~71 ms |

On a sample ML resume the two systems agreed on only 5 of their top 10. The
semantic system surfaced postings such as "Computer Vision / Machine Learning
Engineer" and "Data Scientist – Databricks" that the baseline ranked nowhere.
Session 6 replaces this eyeballing with precision, recall, NDCG, MRR and MAP.

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
| Bag-of-words and TF-IDF vectorization | `src/baseline_tfidf.py` (scikit-learn) |
| N-gram features | `ngram_range=(1, 2)` in the vectorizer |
| Cosine similarity | both retrieval systems |
| Dense transformer embeddings | `src/embedder.py` (Sentence-BERT) |
| Bi-encoder architecture | `all-MiniLM-L6-v2`, siamese fine-tuning |
| Approximate nearest-neighbour search | `src/build_index.py` (FAISS) |
| Two-stage retrieve-and-rank | `src/recommender.py` |

Session 6 adds the evaluation framework: precision@k, recall@k, NDCG@k, MRR and
MAP implemented from scratch, plus an ablation over the re-ranking weights.

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
