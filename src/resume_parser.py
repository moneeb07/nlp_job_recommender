"""Resume understanding: NER, gazetteer-based skill extraction, and field parsing.

Two complementary techniques do the work here:

* spaCy's statistical NER model labels PERSON, ORG, GPE and DATE spans. It was
  trained on OntoNotes, which has no SKILL label — hence the second technique.
* A spaCy PhraseMatcher runs a gazetteer (`data/skills_taxonomy.txt`) over the
  document. It matches *token sequences*, not raw substrings, so "R" matches the
  standalone token "R" but never the "R" inside "React".
"""

import re
from datetime import date

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

from src.config import SKILLS_TAXONOMY, SPACY_MODEL

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*[\w]")

# Deliberately specific patterns. A loose "run of digits" pattern matches year
# ranges and ID numbers, so each alternative below encodes a real phone shape.
_PHONE_RES = [
    re.compile(r"\+\d{1,3}[\s.\-]?\(?\d{1,4}\)?[\s.\-]?\d{3,4}[\s.\-]?\d{3,4}"),
    re.compile(r"\(\d{3}\)\s*\d{3}[\s.\-]?\d{4}"),
    re.compile(r"\b\d{3}[\s.\-]\d{3}[\s.\-]\d{4}\b"),
    re.compile(r"\b0\d{2,3}[\s.\-]?\d{7,8}\b"),
    re.compile(r"\b\d{10,11}\b"),
]

_MONTH_NAMES = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
    "|january|february|march|april|june|july|august"
    "|september|october|november|december"
)
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_DATE_TOKEN = rf"(?:(?:{_MONTH_NAMES})[a-z]*\.?[\s,]+)?(?:19|20)\d{{2}}"
_ONGOING = r"present|current|now|today|ongoing|date"

# Date ranges are read off the raw text rather than off spaCy DATE entities:
# the model splits "Jan 2020 - Present" inconsistently (sometimes one span,
# sometimes two), which loses the pairing that makes a range a duration.
_DATE_RANGE_RE = re.compile(
    rf"({_DATE_TOKEN})\s*(?:-|–|—|to|until|through)\s*({_DATE_TOKEN}|{_ONGOING})\b",
    re.IGNORECASE,
)

# Spelled-out degrees are unambiguous, so these match in any case.
_DEGREE_WORD_RE = re.compile(
    r"\b(?:bachelor(?:'?s)?|master(?:'?s)?|doctorate|doctoral|undergraduate"
    r"|postgraduate|associate degree|diploma|b\.?tech|m\.?tech|mba|ph\.?\s?d)\b",
    re.IGNORECASE,
)

# The two-letter abbreviations must stay case-sensitive: lowercase "be", "me"
# and "ma" are ordinary English words, so a case-insensitive pattern reads
# "this section is to be completed" as a Bachelor of Engineering.
_DEGREE_ABBR_RE = re.compile(r"\b(?:B\.?Sc|M\.?Sc|B\.?S|M\.?S|B\.?A|M\.?A|B\.?E|M\.?E)\b")


def _find_degree(text: str) -> re.Match | None:
    return _DEGREE_WORD_RE.search(text) or _DEGREE_ABBR_RE.search(text)

# "MS" and "BS" are also product prefixes. Without this guard, "MS Office" and
# "MS SQL Server" are both read as master's degrees.
_NOT_A_DEGREE_AFTER = {
    "office", "excel", "word", "powerpoint", "teams", "access", "outlook",
    "sql", "azure", "visio", "project", "sharepoint", "dynamics", "windows",
}

_TITLE_KEYWORDS = {
    "engineer", "developer", "analyst", "scientist", "architect", "administrator",
    "manager", "consultant", "intern", "internship", "researcher", "specialist",
    "designer", "programmer", "director", "officer", "technician", "strategist",
}

_MAX_EDUCATION_CHARS = 300

# Resumes rarely end section headings with a full stop, so spaCy's sentence
# splitter glues the heading onto the first line beneath it.
_SECTION_HEADINGS = {
    "EDUCATION", "EXPERIENCE", "QUALIFICATIONS", "QUALIFICATION", "ACADEMIC",
    "ACADEMICS", "CERTIFICATIONS", "PROJECTS", "SKILLS", "BACKGROUND",
}


def load_skills_taxonomy(path=SKILLS_TAXONOMY) -> list[str]:
    """Read the gazetteer, skipping blank lines and comments."""
    with open(path, encoding="utf-8") as handle:
        lines = (line.strip() for line in handle)
        return [line for line in lines if line and not line.startswith("#")]


def build_skill_matcher(nlp, skills: list[str] | None = None) -> PhraseMatcher:
    """Compile the taxonomy into a PhraseMatcher keyed on lowercase tokens.

    `attr="LOWER"` makes matching case-insensitive while still comparing whole
    tokens, so "Machine Learning" and "machine learning" both match.
    """
    skills = skills if skills is not None else load_skills_taxonomy()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    # nlp.make_doc skips the pipeline: patterns only need tokenization.
    matcher.add("SKILL", [nlp.make_doc(skill) for skill in skills])
    return matcher


def extract_skills(doc: Doc, matcher: PhraseMatcher) -> list[str]:
    """Return the distinct taxonomy skills present in `doc`, alphabetically."""
    found = {doc[start:end].text.lower() for _, start, end in matcher(doc)}
    return sorted(found)


def _extract_email(text: str) -> str:
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else ""


def _extract_phone(text: str) -> str:
    for pattern in _PHONE_RES:
        for match in pattern.finditer(text):
            candidate = match.group(0)
            if 9 <= sum(character.isdigit() for character in candidate) <= 15:
                return candidate.strip()
    return ""


_FORM_LABEL_RE = re.compile(r"^[\w\s]{1,25}:\s*\S")


def _looks_like_a_name(line: str) -> bool:
    words = line.split()
    return (
        1 < len(words) <= 3
        and all(word[:1].isupper() and word.replace(".", "").isalpha() for word in words)
    )


def _extract_name(doc: Doc, text: str) -> str:
    # A name is almost always in the header, so prefer a PERSON found there.
    header = text[:300]
    for entity in doc.ents:
        # A PERSON span running across a line break means spaCy glued two lines
        # together; the result is never a usable name.
        if (
            entity.label_ == "PERSON"
            and entity.start_char < len(header)
            and "\n" not in entity.text
        ):
            return entity.text.strip()

    # spaCy often misses names typed in all caps or with unusual spacing. Only
    # the first line is considered: any later line that merely looks name-shaped
    # is far more likely to be a heading.
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if _looks_like_a_name(first_line):
        return first_line

    for entity in doc.ents:
        if entity.label_ == "PERSON" and "\n" not in entity.text:
            return entity.text.strip()
    return ""


def _parse_date_token(token: str, today: date) -> date | None:
    token = token.strip().lower()
    if re.fullmatch(_ONGOING, token, re.IGNORECASE):
        return today

    year_match = re.search(r"(19|20)\d{2}", token)
    if not year_match:
        return None

    month = 1
    month_match = re.match(rf"({_MONTH_NAMES})", token)
    if month_match:
        month = _MONTHS.get(month_match.group(1), 1)

    return date(int(year_match.group(0)), month, 1)


def _merge_intervals(intervals: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Collapse overlapping date ranges so concurrent roles are not counted twice."""
    merged: list[tuple[date, date]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _extract_years_of_experience(text: str, today: date | None = None) -> float:
    """Total years covered by the date ranges in `text`, overlaps counted once."""
    today = today or date.today()

    intervals = []
    for raw_start, raw_end in _DATE_RANGE_RE.findall(text):
        start = _parse_date_token(raw_start, today)
        end = _parse_date_token(raw_end, today)
        if start and end and end > start:
            intervals.append((start, end))

    months = sum(
        (end.year - start.year) * 12 + (end.month - start.month)
        for start, end in _merge_intervals(intervals)
    )
    return round(months / 12, 1)


def _strip_section_heading(sentence: str) -> str:
    words = sentence.split()
    while words and words[0].strip(":").upper() in _SECTION_HEADINGS:
        words = words[1:]
    return " ".join(words)


def _is_real_degree_match(sentence: str, match: re.Match) -> bool:
    following = sentence[match.end():].strip().split()
    return not (following and following[0].lower().strip(".,") in _NOT_A_DEGREE_AFTER)


def _extract_education(doc: Doc) -> list[str]:
    """Return each sentence that states a degree."""
    found: list[str] = []
    for sentence in doc.sents:
        # Resumes punctuate sparsely, so one "sentence" often spans a heading
        # and the section beneath it. A degree statement occupies a single line
        # in practice, so match line by line: otherwise the whole SKILLS block
        # that follows "BS Computer Science, NUST, 2023" is swallowed with it.
        lines = [line.strip() for line in sentence.text.splitlines() if line.strip()]

        for line in lines or [sentence.text]:
            text = _strip_section_heading(" ".join(line.split()))
            match = _find_degree(text)
            if not match or not _is_real_degree_match(text, match):
                continue

            text = text[:_MAX_EDUCATION_CHARS].strip()
            if text and text not in found:
                found.append(text)
    return found


def _extract_job_titles(text: str) -> list[str]:
    titles: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        # "Intern Name: Moneeb" is a form field, not a job title.
        if not line or len(line) > 100 or _FORM_LABEL_RE.match(line):
            continue
        # Job lines usually read "Data Analyst — Company, Jan 2024 - Present".
        title = re.split(r"\s*[—–\-|,:@]\s*|\s+at\s+", line)[0].strip()

        # The keyword must be in the title itself, not merely somewhere on the
        # line: otherwise form labels like "Intern Name: Moneeb" are harvested.
        if not any(keyword in title.lower() for keyword in _TITLE_KEYWORDS):
            continue

        if title and len(title.split()) <= 5 and title not in titles:
            titles.append(title)
    return titles


def _unique_entity_texts(doc: Doc, label: str, exclude: set[str] = frozenset()) -> list[str]:
    """Distinct entity texts for one label, in document order.

    `exclude` holds the words of the skills already matched. The small spaCy
    model routinely labels technology names as ORG or GPE ("Python" as an
    organization, "NumPy" as a country), so an entity made up entirely of skill
    words is dropped: it is a tool, not an employer or a place.
    """
    seen: list[str] = []
    for entity in doc.ents:
        text = entity.text.strip()
        if entity.label_ != label or not text or text in seen:
            continue

        words = {word for word in re.findall(r"[a-z0-9+#.]+", text.lower())}
        if words and words <= exclude:
            continue

        seen.append(text)
    return seen


class ResumeParser:
    """Parses raw resume text into a structured profile."""

    def __init__(self, model: str = SPACY_MODEL):
        # The full pipeline is kept here: NER supplies the entities and the
        # parser supplies the sentence boundaries that education lookup needs.
        self.nlp = spacy.load(model)
        self.matcher = build_skill_matcher(self.nlp)

    def parse(self, text: str, today: date | None = None) -> dict:
        doc = self.nlp(text)
        skills = extract_skills(doc, self.matcher)
        skill_words = {word for skill in skills for word in skill.split()}

        return {
            "name": _extract_name(doc, text),
            "email": _extract_email(text),
            "phone": _extract_phone(text),
            "skills": skills,
            "organizations": _unique_entity_texts(doc, "ORG", skill_words),
            "locations": _unique_entity_texts(doc, "GPE", skill_words),
            "education": _extract_education(doc),
            "job_titles": _extract_job_titles(text),
            "years_of_experience": _extract_years_of_experience(text, today),
            "entities": [(entity.text, entity.label_) for entity in doc.ents],
            "raw_text": text,
        }


def get_profile_text(profile: dict) -> str:
    """Flatten a parsed profile into the single string that gets embedded.

    Skills, titles and education carry the matching signal; contact details and
    employer names would only add noise, so they are left out.
    """
    parts = profile["skills"] + profile["job_titles"] + profile["education"]
    return " ".join(" ".join(part.split()) for part in parts if part)
