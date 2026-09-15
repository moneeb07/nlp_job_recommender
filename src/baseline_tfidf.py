"""Sparse lexical baseline: TF-IDF vectors compared by cosine similarity.

This is deliberately the simplest thing that works, and it is the number the
neural system in session 5 has to beat.

How it works
------------
**Term frequency (TF)** counts how often a term appears in one document.
**Inverse document frequency (IDF)** scales each term down by how many documents
contain it: `idf(t) = log(N / df(t)) + 1`. A term appearing in every posting
("experience", "team") carries almost no information and gets a weight near
zero, while "kubernetes" appears rarely and so scores highly wherever it does.
The TF-IDF weight of a term in a document is the product of the two.

`sublinear_tf=True` replaces raw TF with `1 + log(tf)`. A term occurring twenty
times is not twenty times more relevant than one occurring once, and the log
dampens that.

Each document becomes a vector over the vocabulary, and similarity is the
**cosine** of the angle between two such vectors:

    cos(a, b) = (a · b) / (||a|| * ||b||)

Cosine is used rather than Euclidean distance because it ignores magnitude: a
long posting and a short resume can still be a perfect match if the *mix* of
terms agrees. Because `TfidfVectorizer` already L2-normalizes every row
(``norm="l2"``), the denominator is 1 and the cosine is just the dot product.

The weakness this baseline exists to demonstrate: TF-IDF matches on **surface
forms only**. A resume saying "neural networks" and a posting saying "deep
learning" share no terms, so they score zero against each other.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    JOBS_PARQUET,
    TFIDF_MATRIX_PATH,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
    TFIDF_VECTORIZER_PATH,
    TOP_K,
)


class NotFittedError(RuntimeError):
    """Raised when a recommender is used before `fit` or `load`."""


class TFIDFRecommender:
    """Ranks job postings against a resume by TF-IDF cosine similarity."""

    def __init__(
        self,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    @property
    def is_fitted(self) -> bool:
        return self.vectorizer is not None and self.matrix is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("Call fit() or load() before using the recommender.")

    def fit(self, job_texts) -> "TFIDFRecommender":
        """Learn the vocabulary and IDF weights, then vectorize every posting.

        `min_df=2` drops terms appearing in only one posting: they cannot help
        match anything and they bloat the vocabulary. `ngram_range=(1, 2)` keeps
        bigrams so that "machine learning" is a feature in its own right rather
        than only the unrelated unigrams "machine" and "learning".
        """
        job_texts = list(job_texts)
        if not job_texts:
            raise ValueError("fit() needs at least one job text.")

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )
        self.matrix = self.vectorizer.fit_transform(job_texts)
        return self

    def score_all(self, resume_text: str) -> np.ndarray:
        """Cosine similarity of `resume_text` against every posting.

        The resume is transformed with the *same* fitted vectorizer, so it lands
        in the same vector space with the same IDF weights. Terms the corpus
        never saw are silently dropped, which is correct: an unseen term carries
        no evidence about any posting.
        """
        self._require_fitted()

        query = self.vectorizer.transform([resume_text or ""])
        # Both sides are L2-normalized by the vectorizer, so this dot product
        # is already the cosine similarity.
        return (self.matrix @ query.T).toarray().ravel()

    def recommend(self, resume_text: str, top_k: int = TOP_K) -> list[dict]:
        """Top `top_k` postings as `{"index", "score"}`, best first."""
        scores = self.score_all(resume_text)
        top_k = min(top_k, len(scores))

        # argpartition finds the top k without sorting all 5,000 scores, then
        # only those k are sorted.
        top = np.argpartition(-scores, top_k - 1)[:top_k] if top_k else []
        ranked = sorted(top, key=lambda index: -scores[index])

        return [{"index": int(index), "score": float(scores[index])} for index in ranked]

    def get_top_terms(self, text: str, n: int = 20) -> list[tuple[str, float]]:
        """The highest-weighted TF-IDF terms in one document."""
        self._require_fitted()

        vector = self.vectorizer.transform([text or ""]).tocoo()
        names = self.vectorizer.get_feature_names_out()
        terms = sorted(
            zip(vector.col, vector.data), key=lambda pair: -pair[1]
        )[:n]
        return [(names[column], float(weight)) for column, weight in terms]

    def shared_terms(
        self, resume_text: str, job_index: int, n: int = 10
    ) -> list[tuple[str, float]]:
        """Terms that actually drove one match, strongest first.

        A term contributes to the cosine score only when it is non-zero on both
        sides, and its contribution is the product of the two weights. Ranking
        by that product answers "why this posting?" far better than listing the
        top terms of either document alone.
        """
        self._require_fitted()

        query = self.vectorizer.transform([resume_text or ""])
        contributions = self.matrix[job_index].multiply(query).tocoo()
        names = self.vectorizer.get_feature_names_out()

        top = sorted(zip(contributions.col, contributions.data), key=lambda p: -p[1])[:n]
        return [(names[column], float(weight)) for column, weight in top]

    def save(
        self,
        vectorizer_path=TFIDF_VECTORIZER_PATH,
        matrix_path=TFIDF_MATRIX_PATH,
    ) -> None:
        self._require_fitted()
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.matrix, matrix_path)

    @classmethod
    def load(
        cls,
        vectorizer_path=TFIDF_VECTORIZER_PATH,
        matrix_path=TFIDF_MATRIX_PATH,
    ) -> "TFIDFRecommender":
        """Restore a fitted model so the app never refits on startup."""
        model = cls()
        model.vectorizer = joblib.load(vectorizer_path)
        model.matrix = joblib.load(matrix_path)
        return model

    @staticmethod
    def is_available(
        vectorizer_path=TFIDF_VECTORIZER_PATH,
        matrix_path=TFIDF_MATRIX_PATH,
    ) -> bool:
        return vectorizer_path.exists() and matrix_path.exists()


def build(text_column: str = "job_text_clean") -> TFIDFRecommender:
    """Fit the baseline over the processed corpus and persist it."""
    jobs = pd.read_parquet(JOBS_PARQUET, columns=[text_column])
    model = TFIDFRecommender().fit(jobs[text_column].fillna("").tolist())
    model.save()
    return model


if __name__ == "__main__":
    print("Fitting TF-IDF baseline…")
    recommender = build()
    rows, features = recommender.matrix.shape
    density = recommender.matrix.nnz / (rows * features)

    print(f"Postings:          {rows:,}")
    print(f"Vocabulary:        {features:,} terms (1-2 grams)")
    print(f"Non-zero weights:  {recommender.matrix.nnz:,} ({density:.3%} dense)")
    print(f"Saved vectorizer:  {TFIDF_VECTORIZER_PATH}")
    print(f"Saved matrix:      {TFIDF_MATRIX_PATH}")
