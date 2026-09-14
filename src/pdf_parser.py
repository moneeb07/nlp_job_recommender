"""Text extraction from resume PDFs."""

import re
from collections import Counter

import pdfplumber

# Only the top/bottom few lines of a page can be a running header or footer.
_EDGE_LINES = 3

# A line must appear at the edge of at least this share of pages to count as
# boilerplate rather than genuine resume content.
_REPEAT_RATIO = 0.6

_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?\d+\s*(?:(?:of|/)\s*\d+)?\s*$", re.IGNORECASE
)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ​]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_repeated_edge_lines(pages: list[str]) -> set[str]:
    if len(pages) < 2:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.split("\n") if line.strip()]
        edges = lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]
        counts.update(set(edges))

    threshold = max(2, round(len(pages) * _REPEAT_RATIO))
    return {line for line, count in counts.items() if count >= threshold}


def _strip_boilerplate(pages: list[str]) -> str:
    repeated = _find_repeated_edge_lines(pages)

    kept_pages = []
    for page in pages:
        lines = [line.strip() for line in page.split("\n")]
        kept = [
            line
            for line in lines
            if line and line not in repeated and not _PAGE_NUMBER_RE.match(line)
        ]
        if kept:
            kept_pages.append("\n".join(kept))

    return "\n\n".join(kept_pages)


def extract_text_from_pdf(file) -> str:
    """Extract cleaned text from a PDF file object.

    Collapses excess whitespace, normalizes line breaks, and drops page numbers
    and headers/footers that repeat across pages.
    """
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]

    pages = [_normalize_whitespace(page) for page in pages]
    return _normalize_whitespace(_strip_boilerplate(pages))
