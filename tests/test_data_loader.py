import pandas as pd
import pytest

from src.data_loader import (
    build_job_text,
    clean_postings,
    extract_min_years_experience,
)


@pytest.mark.parametrize(
    "description, expected",
    [
        ("3+ years of experience required", 3.0),
        ("minimum 2 years experience", 2.0),
        ("at least 5 years of relevant experience", 5.0),
        ("2-4 years of experience", 2.0),
        ("5 or more years of experience in sales", 5.0),
        ("1 year of experience", 1.0),
        ("Experience: 7+ years", 7.0),
    ],
)
def test_experience_requirements_are_parsed(description, expected):
    assert extract_min_years_experience(description) == expected


@pytest.mark.parametrize(
    "description",
    [
        "No requirement stated here.",
        # The number must relate to experience, not to the company's history.
        "We were founded 10 years ago and love our city.",
        "",
    ],
)
def test_unstated_experience_is_none(description):
    assert extract_min_years_experience(description) is None


def _frame(rows):
    return pd.DataFrame(
        rows, columns=["job_id", "title", "company", "location", "description"]
    )


def test_short_descriptions_are_dropped():
    frame = _frame([
        (1, "Data Analyst", "Acme", "Lahore", "x" * 150),
        (2, "Cook", "Diner", "Lahore", "too short"),
    ])

    result = clean_postings(frame, sample_size=None)

    assert list(result["job_id"]) == [1]


def test_duplicate_title_and_company_collapse():
    frame = _frame([
        (1, "Data Analyst", "Acme", "Lahore", "x" * 150),
        (2, "Data Analyst", "Acme", "Karachi", "y" * 150),
        (3, "Data Analyst", "Other", "Lahore", "z" * 150),
    ])

    result = clean_postings(frame, sample_size=None)

    assert len(result) == 2
    assert set(result["company"]) == {"Acme", "Other"}


def test_sampling_is_reproducible():
    frame = _frame([
        (index, f"Role {index}", f"Company {index}", "Lahore", "x" * 150)
        for index in range(50)
    ])

    first = clean_postings(frame, sample_size=10, seed=42)
    second = clean_postings(frame, sample_size=10, seed=42)

    assert len(first) == 10
    assert list(first["job_id"]) == list(second["job_id"])


def test_missing_company_becomes_unknown():
    frame = _frame([(1, "Data Analyst", None, "Lahore", "x" * 150)])

    assert clean_postings(frame, sample_size=None)["company"].iloc[0] == "Unknown"


def test_job_text_truncates_description_but_keeps_metadata():
    row = {
        "title": "Data Analyst",
        "company": "Acme",
        "skills": ["python", "sql"],
        "description": "d" * 5000,
    }

    text = build_job_text(row)

    assert text.startswith("Data Analyst Acme python, sql")
    assert len(text) < 2000


def test_job_text_handles_a_job_with_no_skills():
    row = {"title": "Cook", "company": "Diner", "skills": [], "description": "Cooking."}

    assert build_job_text(row) == "Cook Diner Cooking."
