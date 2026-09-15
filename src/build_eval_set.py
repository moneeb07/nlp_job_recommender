"""Build a labelled evaluation set without manual annotation.

    python -m src.build_eval_set

How the labels are made
-----------------------
Each of `N_QUERIES` postings is treated as a synthetic "resume": its own text
becomes the query. Another posting counts as **relevant** to it when both hold:

1. the two share a normalized job-title family ("Senior ML Engineer (Python)"
   and "Machine Learning Engineer II" both normalize to "machine learning
   engineer"), and
2. their extracted skill sets have Jaccard similarity above `MIN_SKILL_OVERLAP`.

THESE ARE HEURISTIC LABELS, NOT HUMAN RELEVANCE JUDGEMENTS.
-----------------------------------------------------------
Every number computed from this set inherits the following limitations, and any
honest reading of the benchmark has to state them:

* **The labels encode the same notion of similarity the systems are scored on.**
  Title text and extracted skills also feed the retrievers, so this measures
  internal consistency more than true relevance. A system that happened to match
  this heuristic exactly would score perfectly while being useless to a person.
* **Relevance is binary and symmetric.** Real relevance is graded ("perfect fit"
  vs. "worth a look") and depends on the individual, not just the posting.
* **Title families are lexical.** "Data Scientist" and "Machine Learning
  Engineer" are near-identical jobs in practice but never share a family here,
  so genuinely good matches are labelled irrelevant and the absolute scores are
  pessimistic.
* **Only queries with at least one relevant posting are kept**, which biases the
  sample towards crowded title families. Recall is undefined without them, but
  the selection is not random.

The set is therefore useful for *comparing systems against each other* on equal
terms, and not for claiming an absolute quality level. Replacing it with human
judgements is the single biggest improvement available to this project.
"""

import json
import re

import pandas as pd

from src.config import EVAL_DATA_DIR, EVAL_SET_JSON, JOBS_PARQUET, RANDOM_SEED

# Reused rather than reimplemented: the labels must use exactly the same notion
# of skill overlap that the re-ranker scores with, or the two quietly diverge.
from src.recommender import as_skill_set, jaccard_similarity

N_QUERIES = 50
MIN_SKILL_OVERLAP = 0.4

# Seniority and level markers that describe the rung, not the job.
_LEVEL_WORDS = {
    "senior", "sr", "snr", "junior", "jr", "staff", "principal", "lead",
    "chief", "head", "entry", "entry-level", "mid", "mid-level", "associate",
    "i", "ii", "iii", "iv", "v", "1", "2", "3", "4", "5",
}

_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")
_PUNCTUATION_RE = re.compile(r"[^a-z0-9\s+#.-]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Reduce a posting title to its job-title family.

    Drops parentheticals, anything after a separator (usually a location or a
    second title), seniority words, and trailing level numbers.
    """
    title = (title or "").lower()
    title = _PARENTHETICAL_RE.sub(" ", title)

    # Everything after " - ", "," or "/" is almost always a location, a
    # department, or a duplicated title.
    title = re.split(r"\s+-\s+|,|/|\|", title)[0]

    title = _PUNCTUATION_RE.sub(" ", title)
    words = [word for word in _WHITESPACE_RE.split(title) if word]
    # Compare with surrounding dots and dashes stripped so "Sr." matches "sr",
    # while words keep their own punctuation (".net" stays ".net").
    words = [word for word in words if word.strip(".-") not in _LEVEL_WORDS]

    return " ".join(words).strip()


def find_relevant(
    skills: list[set],
    family_members: list[int],
    query_position: int,
    min_overlap: float,
) -> list[int]:
    """Postings in the same title family whose skills overlap enough."""
    query_skills = skills[query_position]
    return [
        position
        for position in family_members
        if position != query_position
        and jaccard_similarity(query_skills, skills[position]) > min_overlap
    ]


def build_eval_set(
    jobs: pd.DataFrame,
    n_queries: int = N_QUERIES,
    min_overlap: float = MIN_SKILL_OVERLAP,
    seed: int = RANDOM_SEED,
) -> dict:
    families = jobs["title"].map(normalize_title)
    skills = [as_skill_set(row) for row in jobs["skills"]]

    members_by_family: dict[str, list[int]] = {}
    for position, family in enumerate(families):
        if family:
            members_by_family.setdefault(family, []).append(position)

    # Only families with more than one posting can produce a relevant pair.
    candidates = [
        position
        for family, members in members_by_family.items()
        if len(members) > 1
        for position in members
    ]

    queries = []
    for position in candidates:
        relevant = find_relevant(
            skills, members_by_family[families[position]], position, min_overlap
        )
        if relevant:
            queries.append(
                {
                    "query_position": position,
                    "job_id": int(jobs.iloc[position]["job_id"]),
                    "title": jobs.iloc[position]["title"],
                    "title_family": families[position],
                    "relevant": relevant,
                }
            )

    eligible = len(queries)
    sampled = (
        pd.Series(range(eligible)).sample(n=n_queries, random_state=seed).tolist()
        if eligible > n_queries
        else list(range(eligible))
    )
    queries = [queries[position] for position in sorted(sampled)]

    relevant_counts = [len(query["relevant"]) for query in queries]
    return {
        "queries": queries,
        "metadata": {
            "n_queries": len(queries),
            "eligible_queries": eligible,
            "corpus_size": len(jobs),
            "min_skill_overlap": min_overlap,
            "seed": seed,
            "job_ids": jobs["job_id"].tolist(),
            "mean_relevant_per_query": (
                sum(relevant_counts) / len(relevant_counts) if relevant_counts else 0.0
            ),
            "labels": "heuristic, not human judgements — see module docstring",
        },
    }


def load_eval_set(path=EVAL_SET_JSON, jobs: pd.DataFrame | None = None) -> dict:
    with open(path, encoding="utf-8") as handle:
        eval_set = json.load(handle)

    # Row positions only mean something against the corpus they were built from.
    if jobs is not None and eval_set["metadata"]["job_ids"] != jobs["job_id"].tolist():
        raise ValueError(
            "The evaluation set was built from a different job table. "
            "Rebuild with `python -m src.build_eval_set`."
        )
    return eval_set


def main() -> None:
    jobs = pd.read_parquet(JOBS_PARQUET, columns=["job_id", "title", "skills"])
    eval_set = build_eval_set(jobs)
    metadata = eval_set["metadata"]

    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SET_JSON, "w", encoding="utf-8") as handle:
        json.dump(eval_set, handle, indent=2)

    counts = [len(query["relevant"]) for query in eval_set["queries"]]
    print(f"Corpus:               {metadata['corpus_size']:,} postings")
    print(f"Title families:       {jobs['title'].map(normalize_title).nunique():,}")
    print(f"Eligible queries:     {metadata['eligible_queries']:,}")
    print(f"Sampled queries:      {metadata['n_queries']}")
    if counts:
        print(f"Relevant per query:   min {min(counts)}, "
              f"median {sorted(counts)[len(counts) // 2]}, max {max(counts)}, "
              f"mean {sum(counts) / len(counts):.1f}")
    print(f"Saved:                {EVAL_SET_JSON}")
    print()
    print("Labels are heuristic, not human judgements. See the module docstring.")


if __name__ == "__main__":
    main()
