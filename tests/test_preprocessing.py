from src.pdf_parser import _normalize_whitespace, _strip_boilerplate
from src.preprocessing import get_token_analysis, preprocess_text

SAMPLE = "The candidate was running   models\n\n\n and   analysing  datasets in 2023."


def test_preprocess_removes_stopwords_and_punctuation():
    cleaned = preprocess_text(SAMPLE)

    assert "the" not in cleaned.split()
    assert "." not in cleaned
    assert "2023" not in cleaned


def test_preprocess_lemmatizes():
    cleaned = preprocess_text(SAMPLE).split()

    assert "run" in cleaned
    assert "running" not in cleaned
    assert "model" in cleaned


def test_preprocess_drops_symbols_but_keeps_skill_names():
    cleaned = preprocess_text("Office Flow | Consultants & Co. C++ .NET 50%").split()

    assert "|" not in cleaned
    assert "&" not in cleaned
    assert "%" not in cleaned
    assert "c++" in cleaned
    assert ".net" in cleaned


def test_preprocess_splits_hyphenated_terms():
    # spaCy tokenizes on hyphens, so "scikit-learn" becomes two tokens. The
    # PhraseMatcher and the (1,2)-gram vectorizer both tokenize the same way,
    # so the term is still recoverable downstream.
    cleaned = preprocess_text("scikit-learn").split()

    assert cleaned == ["scikit", "learn"]


def test_preprocess_handles_empty_text():
    assert preprocess_text("") == ""
    assert get_token_analysis("") == []


def test_token_analysis_shape_and_flags():
    rows = get_token_analysis("The engineer builds models.")

    assert all(len(row) == 4 for row in rows)

    flags = {token: is_stop for token, _, _, is_stop in rows}
    assert flags["The"] is True
    assert flags["engineer"] is False


def test_normalize_whitespace_collapses_runs():
    assert _normalize_whitespace("a  \t b\r\n\n\n\nc  ") == "a b\n\nc"


def test_strip_boilerplate_drops_repeated_headers_and_page_numbers():
    pages = [
        "Moneeb — Resume\nExperience section\nPage 1 of 2",
        "Moneeb — Resume\nEducation section\nPage 2 of 2",
    ]

    result = _strip_boilerplate(pages)

    assert "Moneeb — Resume" not in result
    assert "Page 1 of 2" not in result
    assert "Experience section" in result
    assert "Education section" in result


def test_strip_boilerplate_keeps_single_page_content():
    result = _strip_boilerplate(["Moneeb — Resume\nExperience section"])

    assert "Moneeb — Resume" in result
