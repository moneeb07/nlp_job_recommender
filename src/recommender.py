"""Two-stage retrieve-and-rank recommender.

Stage 1 (**retrieval**) asks FAISS for the `top_n` postings whose embeddings sit
closest to the resume. It is cheap and runs over the whole corpus, but it only
knows about overall semantic similarity.

Stage 2 (**re-ranking**) rescores those few candidates with signals that are too
expensive or too structured to encode in a single vector: how many of the
required skills the candidate actually has, and whether they meet the stated
experience bar.

Splitting the work this way is the standard information-retrieval pattern.
Scoring all 5,000 postings on every signal would waste effort on thousands of
postings that are obviously irrelevant, while retrieval alone cannot tell that a
candidate is two years short of the requirement.

    final = 0.50 * semantic_similarity
          + 0.30 * skill_overlap
          + 0.20 * experience_match

The weights are in `src/config.py`; session 6 tests alternatives with an
ablation study rather than taking them on faith.
"""

import pickle

import faiss
import numpy as np
import pandas as pd

from src.config import (
    FAISS_INDEX_PATH,
    JOBS_META_PATH,
    JOBS_PARQUET,
    RETRIEVE_TOP_N,
    TOP_K,
    WEIGHT_EXPERIENCE,
    WEIGHT_SEMANTIC,
    WEIGHT_SKILL_OVERLAP,
)
from src.embedder import Embedder
from src.resume_parser import get_profile_text


def jaccard_similarity(left: set, right: set) -> float:
    """|A ∩ B| / |A ∪ B|.

    Chosen over raw overlap count because it is already bounded in [0, 1] and it
    penalises a posting that demands twenty skills when the candidate has three,
    rather than rewarding it for the size of the intersection alone.
    """
    if not left or not right:
        return 0.0

    union = left | right
    return len(left & right) / len(union) if union else 0.0


def as_skill_set(value) -> set[str]:
    """Normalize a skills cell into a lowercase set.

    Parquet hands back a numpy array rather than a list, and `array or []`
    raises `ValueError` because an array's truth value is ambiguous. Missing
    values arrive as None or NaN, neither of which is iterable.
    """
    if value is None:
        return set()

    try:
        return {str(skill).lower() for skill in value}
    except TypeError:
        return set()


def experience_match(candidate_years: float, required_years) -> float:
    """How well the candidate meets a posting's stated experience bar.

    1.0 when the posting states no requirement, or when the candidate meets it.
    Otherwise the shortfall scales linearly: three years against a five-year
    requirement scores 0.6.
    """
    if required_years is None or pd.isna(required_years) or required_years <= 0:
        return 1.0

    if candidate_years >= required_years:
        return 1.0

    return float(min(max(candidate_years, 0.0) / required_years, 1.0))


class JobRecommender:
    """Retrieves with FAISS, then re-ranks on skills and experience."""

    def __init__(
        self,
        index: faiss.Index,
        jobs: pd.DataFrame,
        embedder: Embedder | None = None,
        weight_semantic: float = WEIGHT_SEMANTIC,
        weight_skills: float = WEIGHT_SKILL_OVERLAP,
        weight_experience: float = WEIGHT_EXPERIENCE,
    ):
        if index.ntotal != len(jobs):
            raise ValueError(
                f"Index holds {index.ntotal} vectors but the job table has "
                f"{len(jobs)} rows. Rebuild with `python -m src.build_index`."
            )

        self.index = index
        self.jobs = jobs.reset_index(drop=True)
        self.embedder = embedder or Embedder()
        self.weight_semantic = weight_semantic
        self.weight_skills = weight_skills
        self.weight_experience = weight_experience

    @classmethod
    def load(
        cls,
        index_path=FAISS_INDEX_PATH,
        meta_path=JOBS_META_PATH,
        jobs_path=JOBS_PARQUET,
        embedder: Embedder | None = None,
    ) -> "JobRecommender":
        index = faiss.read_index(str(index_path))
        jobs = pd.read_parquet(jobs_path)

        # The index stores no identifiers, so a stale index silently returns
        # rows belonging to a different corpus. Verify the alignment instead.
        with open(meta_path, "rb") as handle:
            metadata = pickle.load(handle)
        if list(metadata["job_ids"]) != jobs["job_id"].tolist():
            raise ValueError(
                "The FAISS index was built from a different job table. "
                "Rebuild with `python -m src.build_index`."
            )

        return cls(index, jobs, embedder=embedder)

    @staticmethod
    def is_available(index_path=FAISS_INDEX_PATH, meta_path=JOBS_META_PATH) -> bool:
        return index_path.exists() and meta_path.exists()

    def retrieve(self, profile: dict, top_n: int = RETRIEVE_TOP_N) -> list[dict]:
        """Stage 1 — nearest postings by embedding similarity."""
        query_text = profile.get("profile_text") or get_profile_text(profile)
        query = self.embedder.encode_single(query_text).reshape(1, -1)

        top_n = min(top_n, self.index.ntotal)
        if top_n == 0:
            return []

        similarities, positions = self.index.search(query, top_n)

        candidates = []
        for similarity, position in zip(similarities[0], positions[0]):
            if position < 0:  # FAISS pads with -1 when it runs out of vectors
                continue
            candidates.append(
                {
                    "index": int(position),
                    # Cosine runs [-1, 1]; a negative value means unrelated, so
                    # clipping at zero keeps the component in [0, 1] without
                    # compressing the useful range the way (x + 1) / 2 would.
                    "semantic_score": float(max(similarity, 0.0)),
                }
            )
        return candidates

    def rerank(self, candidates: list[dict], profile: dict) -> list[dict]:
        """Stage 2 — blend semantic similarity with skills and experience."""
        resume_skills = as_skill_set(profile.get("skills"))
        candidate_years = float(profile.get("years_of_experience") or 0.0)

        ranked = []
        for candidate in candidates:
            job = self.jobs.iloc[candidate["index"]]
            job_skills = as_skill_set(job["skills"])

            overlap = jaccard_similarity(resume_skills, job_skills)
            experience = experience_match(candidate_years, job["min_years_experience"])
            semantic = candidate["semantic_score"]

            final = (
                self.weight_semantic * semantic
                + self.weight_skills * overlap
                + self.weight_experience * experience
            )

            ranked.append(
                {
                    **candidate,
                    "score": float(final),
                    "skill_overlap": float(overlap),
                    "experience_match": float(experience),
                    "matched_skills": sorted(resume_skills & job_skills),
                    "missing_skills": sorted(job_skills - resume_skills),
                }
            )

        ranked.sort(key=lambda result: -result["score"])
        return ranked

    def recommend(
        self,
        profile: dict,
        top_k: int = TOP_K,
        top_n: int = RETRIEVE_TOP_N,
    ) -> list[dict]:
        """Both stages, returning every component sub-score for inspection."""
        return self.rerank(self.retrieve(profile, top_n=top_n), profile)[:top_k]

    def retrieve_only(self, profile: dict, top_k: int = TOP_K) -> list[dict]:
        """Stage 1 alone — the ablation session 6 compares re-ranking against."""
        candidates = self.retrieve(profile, top_n=top_k)
        for candidate in candidates:
            candidate["score"] = candidate["semantic_score"]
        return candidates
