import faiss
import numpy as np
import pandas as pd
import pytest

from src.recommender import (
    JobRecommender,
    as_skill_set,
    experience_match,
    jaccard_similarity,
)


class StubEmbedder:
    """Returns a fixed query vector so similarities are exactly predictable."""

    def __init__(self, vector=(1.0, 0.0)):
        self.vector = np.array(vector, dtype="float32")

    def encode_single(self, text):
        return self.vector


# job 0 points the same way as the query, job 1 is orthogonal, job 2 is opposite
# (negative inner product, which must be clipped to zero).
JOB_VECTORS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype="float32")

JOBS = pd.DataFrame(
    {
        "job_id": [10, 20, 30],
        "title": ["Data Analyst", "Java Developer", "Chef"],
        "company": ["Acme", "Globex", "Bistro"],
        "location": ["Lahore", "Karachi", "Rawalpindi"],
        "skills": [["python", "sql"], ["java"], []],
        "min_years_experience": [None, 10.0, 3.0],
    }
)

PROFILE = {
    "skills": ["Python", "SQL"],
    "years_of_experience": 2.0,
    "job_titles": ["Data Analyst"],
    "education": [],
    "profile_text": "python sql data analyst",
}


@pytest.fixture
def recommender():
    index = faiss.IndexFlatIP(2)
    index.add(JOB_VECTORS)
    return JobRecommender(index, JOBS, embedder=StubEmbedder())


# --- Jaccard ---------------------------------------------------------------

def test_jaccard_of_identical_sets_is_one():
    assert jaccard_similarity({"python", "sql"}, {"python", "sql"}) == 1.0


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard_similarity({"python"}, {"java"}) == 0.0


def test_jaccard_is_intersection_over_union():
    # {a, b} vs {b, c}: one shared, three in the union.
    assert jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_jaccard_penalises_a_long_requirement_list():
    resume = {"python", "sql"}
    short_job = {"python", "sql"}
    long_job = {"python", "sql", "go", "rust", "scala", "perl"}

    assert jaccard_similarity(resume, short_job) > jaccard_similarity(resume, long_job)


@pytest.mark.parametrize("left, right", [(set(), {"python"}), ({"python"}, set()), (set(), set())])
def test_jaccard_handles_empty_sets(left, right):
    assert jaccard_similarity(left, right) == 0.0


# --- Skill cell normalization ----------------------------------------------

def test_skill_set_from_a_numpy_array():
    """Parquet returns arrays, not lists — `array or []` raises ValueError."""
    assert as_skill_set(np.array(["Python", "SQL"])) == {"python", "sql"}


def test_skill_set_from_an_empty_array():
    assert as_skill_set(np.array([], dtype=object)) == set()


@pytest.mark.parametrize("missing", [None, np.nan, float("nan")])
def test_skill_set_from_missing_values(missing):
    assert as_skill_set(missing) == set()


def test_skill_set_lowercases_a_list():
    assert as_skill_set(["Power BI", "AWS"]) == {"power bi", "aws"}


def test_reranking_handles_array_valued_skill_columns():
    """Mirrors the real parquet dtype end to end, not just the helper."""
    index = faiss.IndexFlatIP(2)
    index.add(JOB_VECTORS)
    jobs = JOBS.copy()
    jobs["skills"] = [np.array(["python", "sql"]), np.array(["java"]), np.array([])]
    recommender = JobRecommender(index, jobs, embedder=StubEmbedder())

    results = {r["index"]: r for r in recommender.recommend(PROFILE, top_k=3)}

    assert results[0]["skill_overlap"] == 1.0
    assert results[2]["skill_overlap"] == 0.0


# --- Experience ------------------------------------------------------------

@pytest.mark.parametrize("required", [None, np.nan, 0, 0.0])
def test_no_stated_requirement_scores_full_marks(required):
    assert experience_match(0.0, required) == 1.0


def test_meeting_the_requirement_scores_one():
    assert experience_match(5.0, 3.0) == 1.0


def test_exactly_meeting_the_requirement_scores_one():
    assert experience_match(3.0, 3.0) == 1.0


def test_falling_short_scales_linearly():
    assert experience_match(3.0, 5.0) == pytest.approx(0.6)


def test_no_experience_against_a_requirement_scores_zero():
    assert experience_match(0.0, 4.0) == 0.0


def test_negative_years_are_floored_at_zero():
    assert experience_match(-2.0, 4.0) == 0.0


def test_experience_component_never_exceeds_one():
    assert experience_match(40.0, 1.0) == 1.0


# --- Retrieval -------------------------------------------------------------

def test_index_and_job_table_must_agree(recommender):
    index = faiss.IndexFlatIP(2)
    index.add(JOB_VECTORS)

    with pytest.raises(ValueError, match="Rebuild"):
        JobRecommender(index, JOBS.head(2), embedder=StubEmbedder())


def test_retrieval_ranks_by_cosine_similarity(recommender):
    candidates = recommender.retrieve(PROFILE, top_n=3)

    assert [candidate["index"] for candidate in candidates][0] == 0
    assert candidates[0]["semantic_score"] == pytest.approx(1.0)


def test_negative_similarity_is_clipped_to_zero(recommender):
    scores = {c["index"]: c["semantic_score"] for c in recommender.retrieve(PROFILE, top_n=3)}

    # Job 2 points the opposite way, so its raw inner product is -1.
    assert scores[2] == 0.0
    assert all(score >= 0.0 for score in scores.values())


def test_retrieval_respects_top_n(recommender):
    assert len(recommender.retrieve(PROFILE, top_n=2)) == 2


def test_retrieval_caps_at_corpus_size(recommender):
    assert len(recommender.retrieve(PROFILE, top_n=500)) == 3


def test_retrieval_falls_back_to_building_profile_text(recommender):
    profile = {key: value for key, value in PROFILE.items() if key != "profile_text"}

    assert recommender.retrieve(profile, top_n=1)


# --- Re-ranking ------------------------------------------------------------

def test_hybrid_score_is_the_weighted_sum(recommender):
    results = {r["index"]: r for r in recommender.recommend(PROFILE, top_k=3)}

    # Job 0: semantic 1.0, identical skills (Jaccard 1.0), no stated
    # requirement (1.0) -> 0.5 + 0.3 + 0.2.
    assert results[0]["score"] == pytest.approx(1.0)

    # Job 1: orthogonal (0.0), no shared skills (0.0), 2 years against a
    # 10-year bar (0.2) -> 0.2 * 0.2.
    assert results[1]["score"] == pytest.approx(0.04)


def test_subscores_are_returned_for_inspection(recommender):
    result = recommender.recommend(PROFILE, top_k=1)[0]

    assert set(result) >= {
        "index", "score", "semantic_score", "skill_overlap",
        "experience_match", "matched_skills", "missing_skills",
    }


def test_matched_and_missing_skills_are_reported(recommender):
    results = {r["index"]: r for r in recommender.recommend(PROFILE, top_k=3)}

    assert results[0]["matched_skills"] == ["python", "sql"]
    assert results[0]["missing_skills"] == []
    assert results[1]["missing_skills"] == ["java"]


def test_skill_matching_is_case_insensitive(recommender):
    # The profile carries "Python"/"SQL"; the corpus stores them lowercase.
    assert recommender.recommend(PROFILE, top_k=1)[0]["skill_overlap"] == 1.0


def test_results_are_sorted_by_final_score(recommender):
    scores = [result["score"] for result in recommender.recommend(PROFILE, top_k=3)]

    assert scores == sorted(scores, reverse=True)


def test_recommend_respects_top_k(recommender):
    assert len(recommender.recommend(PROFILE, top_k=2)) == 2


def test_reranking_can_overturn_the_retrieval_order():
    """The point of stage 2: the nearest neighbour is not always the best job."""
    # The query sits slightly closer to job 1, but job 0 matches on skills.
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[0.8, 0.6], [1.0, 0.0]], dtype="float32"))
    jobs = pd.DataFrame(
        {
            "job_id": [1, 2],
            "skills": [["python", "sql"], []],
            "min_years_experience": [None, None],
        }
    )
    recommender = JobRecommender(index, jobs, embedder=StubEmbedder((1.0, 0.0)))

    retrieved = recommender.retrieve(PROFILE, top_n=2)
    reranked = recommender.recommend(PROFILE, top_k=2)

    assert retrieved[0]["index"] == 1
    assert reranked[0]["index"] == 0


def test_weights_are_configurable():
    index = faiss.IndexFlatIP(2)
    index.add(JOB_VECTORS)
    semantic_only = JobRecommender(
        index, JOBS, embedder=StubEmbedder(),
        weight_semantic=1.0, weight_skills=0.0, weight_experience=0.0,
    )

    results = {r["index"]: r for r in semantic_only.recommend(PROFILE, top_k=3)}

    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)


def test_retrieve_only_scores_on_semantics_alone(recommender):
    results = recommender.retrieve_only(PROFILE, top_k=3)

    assert all(r["score"] == r["semantic_score"] for r in results)


def test_a_profile_with_no_skills_still_ranks(recommender):
    profile = {**PROFILE, "skills": [], "years_of_experience": 0.0}
    results = recommender.recommend(profile, top_k=3)

    assert len(results) == 3
    assert all(result["skill_overlap"] == 0.0 for result in results)
