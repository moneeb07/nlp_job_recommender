import json

import numpy as np
import pandas as pd
import pytest

from src.build_eval_set import (
    build_eval_set,
    find_relevant,
    load_eval_set,
    normalize_title,
)


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Senior Machine Learning Engineer", "machine learning engineer"),
        ("Machine Learning Engineer II", "machine learning engineer"),
        ("Staff Data Engineer - Remote", "data engineer"),
        ("Sr. Accountant, Treasury", "accountant"),
        ("Data Scientist (Python, SQL)", "data scientist"),
        ("Lead Software Developer - Secret Clearance", "software developer"),
        ("Registered Nurse - RN - Emergency Room", "registered nurse"),
        ("Principal Product Manager | Remote", "product manager"),
        ("Software Development Engineer 4", "software development engineer"),
    ],
)
def test_title_normalization(title, expected):
    assert normalize_title(title) == expected


def test_seniority_variants_collapse_to_one_family():
    variants = [
        "Senior Data Analyst",
        "Data Analyst II",
        "Junior Data Analyst",
        "Data Analyst (Hybrid)",
        "Staff Data Analyst - Austin, TX",
    ]

    assert {normalize_title(title) for title in variants} == {"data analyst"}


def test_different_jobs_stay_in_different_families():
    assert normalize_title("Data Analyst") != normalize_title("Data Engineer")


def test_normalization_handles_missing_and_empty_titles():
    assert normalize_title("") == ""
    assert normalize_title(None) == ""


def test_normalization_keeps_dotted_technology_names():
    # The level-word check strips surrounding dots, but the word itself keeps them.
    assert normalize_title(".NET Developer") == ".net developer"


# --- relevance --------------------------------------------------------------

def test_relevant_excludes_the_query_itself():
    skills = [{"python", "sql"}, {"python", "sql"}]

    assert find_relevant(skills, [0, 1], query_position=0, min_overlap=0.4) == [1]


def test_relevance_requires_enough_skill_overlap():
    skills = [{"python", "sql"}, {"python", "sql"}, {"java", "spring", "aws", "go"}]

    relevant = find_relevant(skills, [0, 1, 2], query_position=0, min_overlap=0.4)

    assert relevant == [1]


def test_overlap_is_strictly_above_the_threshold():
    # {a, b} vs {b, c} is exactly 1/3, which must not qualify at 1/3.
    skills = [{"a", "b"}, {"b", "c"}]

    assert find_relevant(skills, [0, 1], 0, min_overlap=1 / 3) == []


def test_a_job_with_no_skills_has_no_relevant_matches():
    skills = [set(), {"python"}]

    assert find_relevant(skills, [0, 1], 0, min_overlap=0.4) == []


# --- building ---------------------------------------------------------------

def _jobs(rows):
    return pd.DataFrame(rows, columns=["job_id", "title", "skills"])


def test_relevance_needs_the_same_title_family_too():
    """Matching skills alone is not enough — the families must agree."""
    jobs = _jobs([
        (1, "Data Analyst", ["python", "sql"]),
        (2, "Senior Data Analyst", ["python", "sql"]),
        (3, "Chef de Partie", ["python", "sql"]),
    ])

    eval_set = build_eval_set(jobs, n_queries=10)
    relevant_by_position = {q["query_position"]: q["relevant"] for q in eval_set["queries"]}

    assert relevant_by_position[0] == [1]
    assert 2 not in relevant_by_position


def test_queries_without_any_relevant_posting_are_dropped():
    jobs = _jobs([
        (1, "Data Analyst", ["python", "sql"]),
        (2, "Data Analyst", ["nursing", "triage"]),
    ])

    assert build_eval_set(jobs, n_queries=10)["queries"] == []


def test_singleton_families_produce_no_queries():
    jobs = _jobs([(1, "Chef", ["cooking"]), (2, "Pilot", ["flying"])])

    assert build_eval_set(jobs, n_queries=10)["queries"] == []


def test_sampling_respects_the_requested_count():
    jobs = _jobs([
        (index, "Data Analyst", ["python", "sql"]) for index in range(20)
    ])

    eval_set = build_eval_set(jobs, n_queries=5)

    assert len(eval_set["queries"]) == 5
    assert eval_set["metadata"]["eligible_queries"] == 20


def test_sampling_is_reproducible():
    jobs = _jobs([(index, "Data Analyst", ["python", "sql"]) for index in range(20)])

    first = build_eval_set(jobs, n_queries=5, seed=7)
    second = build_eval_set(jobs, n_queries=5, seed=7)

    assert [q["query_position"] for q in first["queries"]] == [
        q["query_position"] for q in second["queries"]
    ]


def test_numpy_skill_arrays_are_handled():
    jobs = _jobs([
        (1, "Data Analyst", np.array(["python", "sql"])),
        (2, "Senior Data Analyst", np.array(["python", "sql"])),
    ])

    assert build_eval_set(jobs, n_queries=10)["queries"][0]["relevant"] == [1]


def test_metadata_records_the_provenance():
    jobs = _jobs([
        (1, "Data Analyst", ["python", "sql"]),
        (2, "Senior Data Analyst", ["python", "sql"]),
    ])

    metadata = build_eval_set(jobs, n_queries=10)["metadata"]

    assert metadata["corpus_size"] == 2
    assert metadata["job_ids"] == [1, 2]
    assert metadata["min_skill_overlap"] == 0.4
    assert "heuristic" in metadata["labels"]


# --- loading ----------------------------------------------------------------

def test_load_rejects_a_mismatched_corpus(tmp_path):
    jobs = _jobs([
        (1, "Data Analyst", ["python", "sql"]),
        (2, "Senior Data Analyst", ["python", "sql"]),
    ])
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(build_eval_set(jobs, n_queries=10)))

    different = _jobs([(99, "Chef", ["cooking"])])

    with pytest.raises(ValueError, match="Rebuild"):
        load_eval_set(path, jobs=different)


def test_load_accepts_the_matching_corpus(tmp_path):
    jobs = _jobs([
        (1, "Data Analyst", ["python", "sql"]),
        (2, "Senior Data Analyst", ["python", "sql"]),
    ])
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(build_eval_set(jobs, n_queries=10)))

    assert load_eval_set(path, jobs=jobs)["metadata"]["corpus_size"] == 2
