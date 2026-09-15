"""Rule-based explanations for a recommendation. No LLM, no API, no latency.

Everything these sentences need has already been computed by the re-ranker, so
generating them costs nothing and cannot hallucinate: each clause is assembled
from numbers the pipeline produced.

Why readiness uses coverage rather than Jaccard
-----------------------------------------------
The re-ranker scores skill overlap with **Jaccard**, |A ∩ B| / |A ∪ B|, which is
symmetric. That is right for *ranking*: it stops a posting from scoring well
just because it lists thirty skills, two of which the candidate happens to have.

It is wrong for *readiness*. A candidate listing 30 skills who matches all 4 a
posting asks for has Jaccard 4/30 = 0.13 — which would read as "stretch role"
when they are in fact fully qualified. The question a reader is asking is
asymmetric: "what fraction of what *this posting* wants do I have?"

    coverage = |matched skills| / |skills the posting lists|

So coverage drives the label and Jaccard stays the ranking signal.
"""

from src.config import GOOD_FIT_THRESHOLD, STRONG_FIT_THRESHOLD

STRONG_FIT = "strong fit"
GOOD_FIT = "good fit"
STRETCH_ROLE = "stretch role"

MAX_SKILLS_LISTED = 3
MAX_SUGGESTIONS = 3


def skill_coverage(matched: list, missing: list) -> float:
    """Fraction of the posting's listed skills the candidate has."""
    required = len(matched) + len(missing)
    return len(matched) / required if required else 0.0


def readiness_label(coverage: float) -> str:
    """Above 0.6 strong fit, 0.3-0.6 good fit, below 0.3 a stretch."""
    if coverage > STRONG_FIT_THRESHOLD:
        return STRONG_FIT
    if coverage >= GOOD_FIT_THRESHOLD:
        return GOOD_FIT
    return STRETCH_ROLE


def _is_stated(value) -> bool:
    """True when a numeric requirement is present and positive.

    Postings with no stated requirement arrive as NaN from the parquet, not as
    None, and `nan <= 0` is False — so a plain None check silently produces
    "it asks for nan years".
    """
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number > 0  # NaN is the only value != itself


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _skills_clause(matched: list, missing: list) -> str:
    required = len(matched) + len(missing)

    if not required:
        return "this posting lists no specific skills we could extract"

    if not matched:
        return f"you list none of the {required} skills this posting asks for"

    shown = matched[:MAX_SKILLS_LISTED]
    extra = len(matched) - len(shown)
    # "a, b and c and 2 more" reads badly, so the conjunction is dropped from
    # the list itself whenever a "and N more" tail follows it.
    listed = f"{', '.join(shown)} and {extra} more" if extra else _join(shown)

    return f"you have {len(matched)} of its {required} listed skills ({listed})"


def _experience_clause(candidate_years: float, required_years) -> str:
    if not _is_stated(required_years):
        return "it states no experience requirement"

    required_years = float(required_years)
    required = f"{required_years:g}-year"
    if candidate_years >= required_years:
        return f"you meet its {required} experience requirement"

    shortfall = required_years - candidate_years
    return (
        f"it asks for {required_years:g} years and your resume shows "
        f"{candidate_years:g}, about {shortfall:g} short"
    )


def _suggestion(missing: list) -> str:
    if not missing:
        return ""
    return f" Consider learning {_join(missing[:MAX_SUGGESTIONS])}."


def explain(result: dict, candidate_years: float = 0.0, required_years=None) -> dict:
    """Build a readiness label and one explanatory sentence for a result.

    `result` is a row from `JobRecommender.recommend`, so it already carries the
    matched and missing skill lists.
    """
    matched = list(result.get("matched_skills") or [])
    missing = list(result.get("missing_skills") or [])

    coverage = skill_coverage(matched, missing)
    label = readiness_label(coverage)

    sentence = (
        f"{label.capitalize()}: {_skills_clause(matched, missing)}, and "
        f"{_experience_clause(candidate_years, required_years)}."
    )

    return {
        "readiness": label,
        "coverage": coverage,
        "text": sentence + _suggestion(missing),
    }
