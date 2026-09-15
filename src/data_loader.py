"""Load the LinkedIn job postings dataset and prepare it for retrieval.

Run as a script to build `data/processed/jobs.parquet`:

    python -m src.data_loader
"""

import re

import pandas as pd
import spacy

from src.config import (
    JOB_TEXT_DESCRIPTION_CHARS,
    JOBS_PARQUET,
    MIN_DESCRIPTION_CHARS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_POSTINGS_CSV,
    SAMPLE_SIZE,
    SPACY_MODEL,
)
from src.preprocessing import preprocess_texts
from src.resume_parser import build_skill_matcher, extract_skills

# Source column -> the normalized name used everywhere downstream.
_COLUMN_MAP = {
    "job_id": "job_id",
    "title": "title",
    "company_name": "company",
    "location": "location",
    "description": "description",
    "skills_desc": "required_skills",
    "formatted_experience_level": "experience_level",
    "normalized_salary": "salary",
}

# "3+ years of experience", "2-4 years of relevant experience", "minimum 5 years ... experience".
# Requiring the word "experience" nearby stops the pattern from picking up
# unrelated numbers such as "founded 10 years ago".
_YEARS_THEN_EXPERIENCE_RE = re.compile(
    r"(\d{1,2})\s*(?:\+|-\s*\d{1,2}|\s*to\s*\d{1,2})?\s*\+?\s*"
    r"(?:or\s+more\s+)?(?:years?|yrs?)[^.!?]{0,40}?experience",
    re.IGNORECASE,
)

# "experience: 3+ years", "experience of at least 2 years".
_EXPERIENCE_THEN_YEARS_RE = re.compile(
    r"experience[^.!?]{0,40}?(\d{1,2})\s*\+?\s*(?:years?|yrs?)",
    re.IGNORECASE,
)

_MAX_PLAUSIBLE_YEARS = 40


def load_raw_postings(path=RAW_POSTINGS_CSV) -> pd.DataFrame:
    """Read the postings CSV, keeping only the columns used downstream."""
    frame = pd.read_csv(path, usecols=list(_COLUMN_MAP))
    return frame.rename(columns=_COLUMN_MAP)


def clean_postings(
    frame: pd.DataFrame,
    sample_size: int | None = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Drop thin and duplicated postings, then take a reproducible sample."""
    frame = frame.copy()
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["title"] = frame["title"].fillna("").astype(str).str.strip()
    frame["company"] = frame["company"].fillna("Unknown").astype(str).str.strip()
    frame["location"] = frame["location"].fillna("").astype(str).str.strip()

    frame = frame[frame["description"].str.len() >= MIN_DESCRIPTION_CHARS]
    frame = frame[frame["title"] != ""]
    frame = frame.drop_duplicates(subset=["title", "company"])

    if sample_size and len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed)

    return frame.reset_index(drop=True)


def extract_min_years_experience(description: str) -> float | None:
    """Minimum years of experience stated in a posting, or None if unstated."""
    for pattern in (_YEARS_THEN_EXPERIENCE_RE, _EXPERIENCE_THEN_YEARS_RE):
        match = pattern.search(description)
        if match:
            years = int(match.group(1))
            if 0 < years <= _MAX_PLAUSIBLE_YEARS:
                return float(years)
    return None


def build_job_text(row) -> str:
    """The string that gets embedded in session 5.

    The description is truncated because the informative part of a posting is
    at the top; the tail is usually boilerplate about benefits and equal
    opportunity policies.
    """
    skills = ", ".join(row["skills"]) if len(row["skills"]) else ""
    parts = [
        row["title"],
        row["company"],
        skills,
        row["description"][:JOB_TEXT_DESCRIPTION_CHARS],
    ]
    return " ".join(" ".join(str(part).split()) for part in parts if part)


def add_skills(frame: pd.DataFrame) -> pd.DataFrame:
    """Tag each posting with taxonomy skills using the resume-side matcher.

    Sharing the matcher matters: resume skills and job skills must come from one
    vocabulary, or the Jaccard overlap in session 5 compares different alphabets.
    """
    # Only tokenization is needed for a PhraseMatcher, so the pipeline is bare.
    nlp = spacy.load(SPACY_MODEL, disable=["tagger", "parser", "ner", "lemmatizer",
                                           "attribute_ruler", "tok2vec"])
    matcher = build_skill_matcher(nlp)

    # Scan the whole description, not the truncated embedding text: requirement
    # lists usually sit well past the first 1500 characters, and truncating here
    # left 47% of postings with no skills at all versus 19% when scanning in full.
    frame["skills"] = [
        extract_skills(doc, matcher)
        for doc in nlp.pipe(frame["description"].tolist(), batch_size=256)
    ]
    return frame


def build_dataset(
    sample_size: int | None = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> pd.DataFrame:
    """Full pipeline: load, clean, enrich, and persist the job corpus."""
    frame = clean_postings(load_raw_postings(), sample_size=sample_size, seed=seed)

    frame = add_skills(frame)
    frame["min_years_experience"] = frame["description"].map(extract_min_years_experience)

    # Raw and cleaned text are kept side by side: the raw text feeds the
    # sentence-transformer, the cleaned text feeds the TF-IDF baseline.
    frame["job_text"] = frame.apply(build_job_text, axis=1)
    frame["job_text_clean"] = preprocess_texts(frame["job_text"].tolist())

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(JOBS_PARQUET, index=False)

    return frame


def summarize(frame: pd.DataFrame) -> str:
    skill_counts = frame["skills"].map(len)
    stated = frame["min_years_experience"].notna()

    lines = [
        f"Jobs:                  {len(frame):,}",
        f"Companies:             {frame['company'].nunique():,}",
        f"Locations:             {frame['location'].nunique():,}",
        f"Median description:    {int(frame['description'].str.len().median()):,} chars",
        f"Skills per job (mean): {skill_counts.mean():.1f}",
        f"Jobs with no skills:   {(skill_counts == 0).sum():,}",
        f"Experience stated:     {stated.sum():,} ({stated.mean():.0%})",
        f"Salary present:        {frame['salary'].notna().sum():,}",
        f"Saved to:              {JOBS_PARQUET}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("Building job dataset…")
    jobs = build_dataset()
    print(summarize(jobs))
