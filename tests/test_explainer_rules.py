import numpy as np
import pytest

from src.explainer_rules import (
    GOOD_FIT,
    STRETCH_ROLE,
    STRONG_FIT,
    explain,
    readiness_label,
    skill_coverage,
)


# --- coverage and labels ----------------------------------------------------

def test_coverage_is_matched_over_required():
    assert skill_coverage(["python", "sql", "aws"], ["go"]) == pytest.approx(0.75)


def test_coverage_of_a_posting_with_no_skills_is_zero():
    assert skill_coverage([], []) == 0.0


def test_coverage_is_one_when_nothing_is_missing():
    assert skill_coverage(["python"], []) == 1.0


@pytest.mark.parametrize(
    "coverage, expected",
    [
        (1.0, STRONG_FIT),
        (0.75, STRONG_FIT),
        (0.61, STRONG_FIT),
        (0.6, GOOD_FIT),      # boundary: 0.3-0.6 inclusive is a good fit
        (0.45, GOOD_FIT),
        (0.3, GOOD_FIT),      # boundary
        (0.29, STRETCH_ROLE),
        (0.0, STRETCH_ROLE),
    ],
)
def test_readiness_thresholds(coverage, expected):
    assert readiness_label(coverage) == expected


def test_a_broad_resume_is_not_punished_for_being_broad():
    """The reason readiness uses coverage instead of the Jaccard used to rank.

    A candidate listing 30 skills who has all 4 the posting asks for scores
    Jaccard 4/30 = 0.13, which would read as a stretch role. Coverage says 1.0.
    """
    matched = [f"skill{index}" for index in range(4)]
    result = {"matched_skills": matched, "missing_skills": []}

    assert explain(result)["readiness"] == STRONG_FIT


# --- sentences --------------------------------------------------------------

def test_strong_match_sentence_matches_the_intended_shape():
    result = {
        "matched_skills": ["python", "sql", "scikit-learn"],
        "missing_skills": ["tensorflow"],
    }

    explanation = explain(result, candidate_years=3.0, required_years=2.0)

    assert explanation["readiness"] == STRONG_FIT
    assert "you have 3 of its 4 listed skills" in explanation["text"]
    assert "python, sql and scikit-learn" in explanation["text"]
    assert "meet its 2-year experience requirement" in explanation["text"]
    assert explanation["text"].endswith("Consider learning tensorflow.")


def test_shortfall_in_experience_is_stated_plainly():
    result = {"matched_skills": ["python"], "missing_skills": []}

    text = explain(result, candidate_years=2.0, required_years=5.0)["text"]

    assert "asks for 5 years" in text
    assert "your resume shows 2" in text
    assert "3 short" in text


def test_unstated_requirement_is_said_so():
    result = {"matched_skills": ["python"], "missing_skills": []}

    for required in (None, 0, 0.0):
        assert "states no experience requirement" in explain(result, 0.0, required)["text"]


def test_nan_requirement_is_treated_as_unstated():
    """Postings with no requirement arrive as NaN from parquet, not None.

    `nan <= 0` is False, so a plain None check produced "it asks for nan years".
    """
    result = {"matched_skills": ["python"], "missing_skills": []}

    for required in (float("nan"), np.nan, np.float64("nan")):
        text = explain(result, candidate_years=4.2, required_years=required)["text"]
        assert "states no experience requirement" in text
        assert "nan" not in text


def test_numpy_floats_are_handled_like_plain_floats():
    result = {"matched_skills": ["python"], "missing_skills": []}

    text = explain(result, candidate_years=3.0, required_years=np.float64(2.0))["text"]

    assert "meet its 2-year experience requirement" in text


def test_no_matched_skills_reads_correctly():
    result = {"matched_skills": [], "missing_skills": ["go", "rust"]}

    explanation = explain(result, candidate_years=1.0, required_years=1.0)

    assert explanation["readiness"] == STRETCH_ROLE
    assert "you list none of the 2 skills" in explanation["text"]


def test_posting_with_no_extractable_skills():
    result = {"matched_skills": [], "missing_skills": []}

    text = explain(result)["text"]

    assert "lists no specific skills" in text
    assert "Consider learning" not in text


def test_long_skill_lists_are_truncated():
    matched = ["a", "b", "c", "d", "e"]
    result = {"matched_skills": matched, "missing_skills": []}

    text = explain(result)["text"]

    assert "a, b, c and 2 more" in text


def test_suggestions_are_capped():
    result = {
        "matched_skills": ["python"],
        "missing_skills": ["go", "rust", "scala", "perl", "haskell"],
    }

    text = explain(result)["text"]

    assert "Consider learning go, rust and scala." in text
    assert "perl" not in text


def test_no_suggestion_when_nothing_is_missing():
    result = {"matched_skills": ["python"], "missing_skills": []}

    assert "Consider learning" not in explain(result)["text"]


def test_single_skill_reads_without_a_conjunction():
    result = {"matched_skills": ["python"], "missing_skills": ["go"]}

    text = explain(result)["text"]

    assert "(python)" in text
    assert "Consider learning go." in text


def test_missing_keys_do_not_crash():
    explanation = explain({})

    assert explanation["readiness"] == STRETCH_ROLE
    assert explanation["text"]


def test_none_skill_lists_are_tolerated():
    explanation = explain({"matched_skills": None, "missing_skills": None})

    assert explanation["coverage"] == 0.0


def test_fractional_years_render_without_trailing_zeros():
    result = {"matched_skills": ["python"], "missing_skills": []}

    text = explain(result, candidate_years=1.5, required_years=2.5)["text"]

    assert "2.5 years" in text
    assert "1.5" in text
    assert "2.50" not in text


def test_every_sentence_ends_with_punctuation():
    cases = [
        {"matched_skills": ["python"], "missing_skills": ["go"]},
        {"matched_skills": [], "missing_skills": []},
        {"matched_skills": [], "missing_skills": ["go"]},
    ]

    for result in cases:
        assert explain(result)["text"].rstrip().endswith(".")
