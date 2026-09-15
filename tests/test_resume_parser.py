from datetime import date

import pytest

from src.resume_parser import (
    ResumeParser,
    _extract_years_of_experience,
    build_skill_matcher,
    extract_skills,
    get_profile_text,
    load_skills_taxonomy,
)

SAMPLE_RESUME = """Ahmed Khan
ahmed.khan@email.com | +92-300-1234567 | Rawalpindi, Pakistan

EXPERIENCE
Data Analyst — Systems Limited, Jan 2024 - Present
Built dashboards using Python and SQL. Worked on machine learning
models with scikit-learn for customer churn prediction.

EDUCATION
BS Computer Science, NUST, 2023
"""

TODAY = date(2026, 9, 15)


@pytest.fixture(scope="module")
def parser():
    return ResumeParser()


@pytest.fixture(scope="module")
def profile(parser):
    return parser.parse(SAMPLE_RESUME, today=TODAY)


def test_profile_has_every_expected_field(profile):
    expected = {
        "name", "email", "phone", "skills", "organizations", "locations",
        "education", "job_titles", "years_of_experience", "entities", "raw_text",
    }
    assert expected <= set(profile)


def test_contact_details(profile):
    assert profile["name"] == "Ahmed Khan"
    assert profile["email"] == "ahmed.khan@email.com"
    assert profile["phone"] == "+92-300-1234567"


def test_skills_include_multi_word_entries(profile):
    assert "machine learning" in profile["skills"]
    assert "python" in profile["skills"]
    assert "sql" in profile["skills"]
    assert "scikit-learn" in profile["skills"]


def test_phrasematcher_does_not_match_inside_words(parser):
    # The reason for a PhraseMatcher over substring search: "React" contains
    # the substring "R", but token-sequence matching never fires on a sub-token.
    # "r" is passed in explicitly because the shipped taxonomy drops bare
    # single letters, and this property is what makes that choice safe to revisit.
    matcher = build_skill_matcher(parser.nlp, ["r", "react"])
    doc = parser.nlp("Built the frontend in React and Redux.")
    skills = extract_skills(doc, matcher)

    assert "react" in skills
    assert "r" not in skills


def test_taxonomy_excludes_ambiguous_single_letters():
    # Measured against the corpus, these matched "b/c", "Class C" and
    # "the go-to person" far more often than the languages themselves.
    skills = set(load_skills_taxonomy())

    assert {"c", "r", "go"}.isdisjoint(skills)
    assert "c programming" in skills
    assert "golang" in skills


def test_taxonomy_loads_and_skips_comments():
    skills = load_skills_taxonomy()

    assert len(skills) > 400
    assert not any(skill.startswith("#") for skill in skills)
    assert "machine learning" in skills


def test_matcher_is_case_insensitive(parser):
    doc = parser.nlp("Experience with MACHINE LEARNING and Natural Language Processing")
    skills = extract_skills(doc, parser.matcher)

    assert "machine learning" in skills
    assert "natural language processing" in skills


def test_entities_are_returned_with_labels(profile):
    labels = {label for _, label in profile["entities"]}

    assert profile["entities"]
    assert "GPE" in labels


def test_locations_come_from_gpe_entities(profile):
    assert "Rawalpindi" in profile["locations"]


def test_tool_names_are_not_reported_as_organizations(parser):
    # en_core_web_sm labels plenty of technology names as ORG or GPE.
    text = "Skills: Python, PyTorch, NumPy, Tableau. Worked at NUST in Pakistan."
    profile = parser.parse(text, today=TODAY)

    assert "Python" not in profile["organizations"]
    assert "NumPy" not in profile["locations"]
    # The raw entity list stays untouched so the UI can show every NER hit.
    assert len(profile["entities"]) >= len(profile["organizations"])


def test_education_sentence_is_captured(profile):
    assert any("Computer Science" in line for line in profile["education"])
    # The section heading should not be glued onto the front.
    assert not any(line.startswith("EDUCATION") for line in profile["education"])


def test_education_does_not_swallow_the_next_section(parser):
    # Resumes rarely end a heading with a full stop, so spaCy runs the degree
    # line and the section beneath it into one "sentence".
    text = "EDUCATION\nBS Computer Science, NUST, 2023\nSKILLS\nPython, SQL, Docker, AWS"
    profile = parser.parse(text, today=TODAY)

    assert profile["education"] == ["BS Computer Science, NUST, 2023"]


def test_job_title_is_captured(profile):
    assert "Data Analyst" in profile["job_titles"]


def test_years_of_experience_from_open_ended_range(profile):
    # Jan 2024 to Sep 2026 is 32 months.
    assert profile["years_of_experience"] == pytest.approx(2.7, abs=0.1)


def test_years_of_experience_plain_year_range():
    assert _extract_years_of_experience("2019-2022", TODAY) == 3.0


def test_overlapping_roles_are_not_double_counted():
    # Two concurrent roles spanning Jun 2020 - Jan 2023 is 31 months, not 55.
    text = "Jun 2020 - Jun 2022 and Jan 2021 - Jan 2023"

    assert _extract_years_of_experience(text, TODAY) == pytest.approx(2.6, abs=0.1)


def test_no_dates_gives_zero_experience():
    assert _extract_years_of_experience("No dates here at all.", TODAY) == 0.0


def test_ms_product_is_not_read_as_a_degree(parser):
    profile = parser.parse("Proficient in MS Office and MS SQL Server.", today=TODAY)

    assert profile["education"] == []


def test_lowercase_be_is_not_read_as_a_degree(parser):
    profile = parser.parse(
        "This section is to be completed by the faculty supervisor.", today=TODAY
    )

    assert profile["education"] == []


def test_form_label_is_not_read_as_a_job_title(parser):
    profile = parser.parse("Intern Name: Moneeb Rahman", today=TODAY)

    assert profile["job_titles"] == []


def test_profile_text_contains_signal_not_contact_details(profile):
    text = get_profile_text(profile)

    assert "machine learning" in text
    assert "Data Analyst" in text
    assert profile["email"] not in text


def test_matcher_can_be_built_for_a_bare_pipeline(parser):
    # Session 3 reuses this matcher over job descriptions with a cheap pipeline.
    matcher = build_skill_matcher(parser.nlp, ["python", "power bi"])
    doc = parser.nlp("We use Python and Power BI.")

    assert extract_skills(doc, matcher) == ["power bi", "python"]


def test_job_titles_are_found_outside_technology(parser):
    """The keyword list was tech-only, so a nursing CV extracted no titles at
    all and had nothing but soft skills left to match on."""
    text = (
        "Sara Ahmed\n"
        "EXPERIENCE\n"
        "Registered Nurse - Shaukat Khanum Hospital, Mar 2021 - Present\n"
        "Staff Nurse - Services Hospital, Jun 2018 - Feb 2021\n"
    )
    profile = parser.parse(text, today=TODAY)

    assert "Registered Nurse" in profile["job_titles"]
    assert "Staff Nurse" in profile["job_titles"]


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Financial Analyst - Engro, Jan 2020 - Present", "Financial Analyst"),
        ("Chef de Partie - Bistro, 2019 - 2021", "Chef de Partie"),
        ("Attorney - Legal Aid, 2018 - 2020", "Attorney"),
        ("Teacher - City School, 2017 - 2019", "Teacher"),
    ],
)
def test_titles_across_sectors(parser, line, expected):
    profile = parser.parse(f"EXPERIENCE\n{line}\n", today=TODAY)

    assert expected in profile["job_titles"]


def test_a_degree_line_is_not_a_job_title(parser):
    # "BS Software Engineering" contains "engineer" but is education.
    profile = parser.parse("EDUCATION\nBS Software Engineering, NUST, 2020\n", today=TODAY)

    assert profile["job_titles"] == []
