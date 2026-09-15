import numpy as np
import pytest

from src.baseline_tfidf import NotFittedError, TFIDFRecommender

CORPUS = [
    "python sql machine learning scikit learn data analysis",
    "java spring boot microservices kubernetes docker backend",
    "python pytorch deep learning computer vision neural network",
    "nurse patient care hospital clinical shift rota",
    "python pandas numpy data analysis reporting dashboard",
]


@pytest.fixture
def model():
    # min_df=1 on this toy corpus: the production default of 2 would drop every
    # term that appears in only one document, leaving almost no vocabulary.
    return TFIDFRecommender(min_df=1).fit(CORPUS)


def test_unfitted_model_refuses_to_score():
    with pytest.raises(NotFittedError):
        TFIDFRecommender().recommend("python")


def test_unfitted_model_refuses_to_save():
    with pytest.raises(NotFittedError):
        TFIDFRecommender().save()


def test_fit_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        TFIDFRecommender().fit([])


def test_matrix_shape_matches_corpus(model):
    assert model.matrix.shape[0] == len(CORPUS)
    assert model.is_fitted


def test_vocabulary_contains_bigrams(model):
    vocabulary = set(model.vectorizer.get_feature_names_out())

    assert "machine learning" in vocabulary
    assert "deep learning" in vocabulary


def test_a_document_is_most_similar_to_itself(model):
    scores = model.score_all(CORPUS[3])

    assert np.argmax(scores) == 3
    assert scores[3] == pytest.approx(1.0, abs=1e-6)


def test_unrelated_documents_score_zero(model):
    # The clinical posting shares no vocabulary with a Java backend query.
    scores = model.score_all("java spring boot microservices")

    assert scores[3] == pytest.approx(0.0, abs=1e-9)


def test_results_are_sorted_by_descending_score(model):
    results = model.recommend("python data analysis", top_k=5)
    scores = [result["score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_recommend_returns_requested_number(model):
    assert len(model.recommend("python", top_k=3)) == 3


def test_recommend_caps_at_corpus_size(model):
    assert len(model.recommend("python", top_k=999)) == len(CORPUS)


def test_recommend_returns_valid_indices(model):
    for result in model.recommend("python", top_k=5):
        assert 0 <= result["index"] < len(CORPUS)


def test_empty_query_scores_zero_everywhere(model):
    scores = model.score_all("")

    assert np.allclose(scores, 0.0)


def test_query_of_unseen_terms_scores_zero(model):
    # Terms absent from the fitted vocabulary carry no evidence.
    scores = model.score_all("zymurgyphlogiston quixotic")

    assert np.allclose(scores, 0.0)


def test_baseline_cannot_match_synonyms(model):
    """The limitation that justifies session 5.

    Document 2 is about deep learning for computer vision. A query sharing its
    surface forms matches. A query meaning the same thing in different words
    scores exactly zero, because TF-IDF compares tokens and not meaning.
    """
    matches_surface_form = model.score_all("neural network computer vision")[2]
    matches_synonym_only = model.score_all("convolutional architectures image recognition")[2]

    assert matches_surface_form > 0
    assert matches_synonym_only == pytest.approx(0.0, abs=1e-9)


def test_top_terms_are_ranked_and_in_vocabulary(model):
    terms = model.get_top_terms(CORPUS[1], n=5)
    weights = [weight for _, weight in terms]
    vocabulary = set(model.vectorizer.get_feature_names_out())

    assert len(terms) == 5
    assert weights == sorted(weights, reverse=True)
    assert all(term in vocabulary for term, _ in terms)
    assert all(weight > 0 for weight in weights)


def test_top_terms_of_an_empty_document_is_empty(model):
    assert model.get_top_terms("", n=5) == []


def test_shared_terms_appear_in_both_documents(model):
    query = "python pytorch deep learning"
    terms = [term for term, _ in model.shared_terms(query, job_index=2, n=10)]

    assert terms
    for term in terms:
        assert term in query
        assert all(word in CORPUS[2] for word in term.split())


def test_shared_terms_are_empty_when_nothing_overlaps(model):
    assert model.shared_terms("java spring boot", job_index=3) == []


def test_shared_terms_are_ranked_by_contribution(model):
    weights = [weight for _, weight in model.shared_terms("python pytorch", 2, n=10)]

    assert weights == sorted(weights, reverse=True)


def test_save_and_load_round_trip(tmp_path, model):
    vectorizer_path = tmp_path / "vectorizer.joblib"
    matrix_path = tmp_path / "matrix.joblib"
    model.save(vectorizer_path, matrix_path)

    restored = TFIDFRecommender.load(vectorizer_path, matrix_path)

    assert restored.is_fitted
    assert np.allclose(restored.score_all("python data"), model.score_all("python data"))


def test_is_available_reflects_files_on_disk(tmp_path, model):
    vectorizer_path = tmp_path / "vectorizer.joblib"
    matrix_path = tmp_path / "matrix.joblib"

    assert not TFIDFRecommender.is_available(vectorizer_path, matrix_path)

    model.save(vectorizer_path, matrix_path)

    assert TFIDFRecommender.is_available(vectorizer_path, matrix_path)


def test_idf_downweights_ubiquitous_terms():
    """A term in every document is uninformative and must weigh less."""
    corpus = ["python common", "java common", "ruby common", "rust common"]
    model = TFIDFRecommender(min_df=1).fit(corpus)

    weights = dict(model.get_top_terms(corpus[0], n=10))

    assert weights["python"] > weights["common"]
