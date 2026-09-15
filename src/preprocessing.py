"""spaCy text preprocessing: tokenization, lemmatization, stopword and punctuation removal."""

import spacy

from src.config import SPACY_MODEL

# The parser and NER are not needed for lemmatization and cost most of the runtime.
# Session 2 loads its own pipeline with NER enabled for entity extraction.
try:
    NLP = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
except OSError as error:
    raise OSError(
        f"spaCy model '{SPACY_MODEL}' is not installed. "
        f"Run: python -m spacy download {SPACY_MODEL}"
    ) from error


def _carries_signal(lemma: str) -> bool:
    """True if the lemma has at least one alphanumeric character.

    spaCy tags stray symbols like "|" as NOUN rather than PUNCT, so `is_punct`
    alone lets them through. Checking for an alphanumeric character drops those
    while keeping skill names such as "C++", ".NET", and "scikit-learn".
    """
    return any(character.isalnum() for character in lemma)


def _clean_doc(doc) -> str:
    lemmas = [
        token.lemma_.lower().strip()
        for token in doc
        if not (token.is_stop or token.is_punct or token.is_space or token.like_num)
    ]
    return " ".join(lemma for lemma in lemmas if _carries_signal(lemma))


def preprocess_text(text: str) -> str:
    """Return `text` as a cleaned, space-joined string of lowercase lemmas.

    Stopwords, punctuation, whitespace, and standalone numbers are removed.
    """
    if not text:
        return ""

    return _clean_doc(NLP(text))


def preprocess_texts(texts, batch_size: int = 64) -> list[str]:
    """Batch version of `preprocess_text` for whole corpora.

    `nlp.pipe` batches documents through the pipeline, which is far faster than
    calling the model once per document.
    """
    return [_clean_doc(doc) for doc in NLP.pipe(texts, batch_size=batch_size)]


def get_token_analysis(text: str) -> list[tuple[str, str, str, bool]]:
    """Return (token, lemma, POS, is_stopword) for every non-whitespace token.

    Kept verbose on purpose: the UI renders this to make the pipeline visible.
    """
    if not text:
        return []

    doc = NLP(text)
    return [
        (token.text, token.lemma_, token.pos_, token.is_stop)
        for token in doc
        if not token.is_space
    ]
