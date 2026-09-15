import numpy as np
import pytest

from src.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_single_vector_has_the_model_dimension(embedder):
    vector = embedder.encode_single("python data engineer")

    assert vector.shape == (embedder.dimension,)
    assert vector.dtype == np.float32


def test_vectors_are_unit_length(embedder):
    """Normalization is what lets FAISS use inner product as cosine."""
    vector = embedder.encode_single("machine learning engineer")

    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_batch_returns_one_row_per_text(embedder):
    vectors = embedder.encode_texts(["python", "java", "sql"], show_progress_bar=False)

    assert vectors.shape == (3, embedder.dimension)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_empty_batch_keeps_the_expected_shape(embedder):
    vectors = embedder.encode_texts([], show_progress_bar=False)

    assert vectors.shape == (0, embedder.dimension)


def test_none_and_empty_text_do_not_crash(embedder):
    vectors = embedder.encode_texts(["", None], show_progress_bar=False)

    assert vectors.shape == (2, embedder.dimension)


def test_encoding_is_deterministic(embedder):
    first = embedder.encode_single("data scientist with sql")
    second = embedder.encode_single("data scientist with sql")

    assert np.allclose(first, second)


def test_inner_product_of_a_vector_with_itself_is_one(embedder):
    vector = embedder.encode_single("kubernetes platform engineer")

    assert float(vector @ vector) == pytest.approx(1.0, abs=1e-5)


def test_batch_and_single_encoding_agree(embedder):
    text = "senior backend developer"
    single = embedder.encode_single(text)
    batched = embedder.encode_texts([text], show_progress_bar=False)[0]

    assert np.allclose(single, batched, atol=1e-6)


def test_related_texts_score_higher_than_unrelated(embedder):
    query = embedder.encode_single("machine learning engineer building models")
    related = embedder.encode_single("data scientist training predictive models")
    unrelated = embedder.encode_single("line cook preparing food in a restaurant")

    assert float(query @ related) > float(query @ unrelated)


def test_embeddings_match_synonyms_that_share_no_words(embedder):
    """The capability TF-IDF lacks.

    These two phrases have no token in common, so the sparse baseline scores
    them at exactly zero (see test_baseline_cannot_match_synonyms). A dense
    bi-encoder should still place them close together.
    """
    left = embedder.encode_single("deep learning neural networks")
    right = embedder.encode_single("convolutional architectures for image recognition")

    assert set("deep learning neural networks".split()).isdisjoint(
        "convolutional architectures for image recognition".split()
    )
    assert float(left @ right) > 0.3
