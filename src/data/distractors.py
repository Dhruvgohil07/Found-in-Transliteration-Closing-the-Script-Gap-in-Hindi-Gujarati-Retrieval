"""Distractor passages from a Wikipedia dump, for a collection that is too small (Phase 10).

IndicQA-gu judges only 248 passages, so retrieval over it alone is close to
trivial. We add unjudged Gujarati Wikipedia passages around them, in three steps:

1. ``clean_wiki_text``: the dump still has ``<ref>`` footnotes and other HTML
   tags; they are removed and spaces collapsed.
2. ``overlapping_articles``: the judged passages are themselves taken from
   Wikipedia. An article that shares any word 8-gram with a judged passage
   probably contains a copy of it (or of the answer), which would sit in the
   collection unjudged and count as wrong. Such articles are dropped whole.
3. ``pack_passages``: the dump's lines are mostly headings and short list items,
   while the judged passages are long (median 136 words). If distractors were
   much shorter, passage length alone would point to the right answer. Lines of
   at least ``min_line_words`` words are packed, in order, into passages of
   ``min_words``..``max_words`` words; a longer line is first split at sentence
   ends.
4. The build script then drops mostly-English passages (``latin_share``) and
   exact repeats (templated village stubs).

Nothing here downloads data; ``scripts/build_collection_gu.py`` does that.
"""

import re
from collections.abc import Iterable, Iterator

# <ref ...>...</ref> (possibly over several lines) and self-closing <ref .../>.
_REF = re.compile(r"<ref[^>/]*?(?:/>|>.*?</ref\s*>)", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^<>\n]{1,200}>")          # any other leftover HTML tag
_SPACES = re.compile(r"[^\S\n]+")                # whitespace except newlines
# A sentence ends at . ! ? or the danda (।), followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def clean_wiki_text(text: str) -> str:
    """Remove footnotes and HTML tags; keep line breaks, collapse other spaces."""
    text = _REF.sub(" ", text)
    text = _TAG.sub(" ", text)
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def word_ngrams(text: str, n: int) -> set[str]:
    """All runs of ``n`` consecutive whitespace-separated words, joined by one space."""
    words = text.split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def overlapping_articles(articles: dict[str, str], protected: Iterable[str], n: int = 8,
                         min_shared: int = 1) -> dict[str, int]:
    """Articles sharing at least ``min_shared`` word ``n``-grams with any protected text.

    Returns ``{article_id: number of shared n-grams}``.
    """
    protected_grams: set[str] = set()
    for text in protected:
        protected_grams |= word_ngrams(text, n)
    hits = {}
    for article_id, text in articles.items():
        shared = len(word_ngrams(text, n) & protected_grams)
        if shared >= min_shared:
            hits[article_id] = shared
    return hits


def _split_long(line: str, max_words: int) -> list[list[str]]:
    """Split one line into pieces of at most ``max_words`` words, at sentence ends
    where possible (a single sentence longer than that is cut into word windows)."""
    pieces: list[list[str]] = []
    current: list[str] = []
    for sentence in _SENTENCE_END.split(line):
        words = sentence.split()
        while len(words) > max_words:              # one very long sentence
            if current:
                pieces.append(current)
                current = []
            pieces.append(words[:max_words])
            words = words[max_words:]
        if current and len(current) + len(words) > max_words:
            pieces.append(current)
            current = []
        current += words
    if current:
        pieces.append(current)
    return pieces


def pack_passages(text: str, min_words: int = 100, max_words: int = 250,
                  min_line_words: int = 5, min_last_words: int = 30) -> list[str]:
    """Cut one cleaned article into passages of about ``min_words``..``max_words`` words.

    Lines shorter than ``min_line_words`` (headings, short list items) are skipped.
    Lines are added to the current passage until it reaches ``min_words``; a line
    that would push it over ``max_words`` starts a new passage instead (the
    unfinished one is kept if it has at least ``min_last_words`` words). What is
    left at the end of the article joins the previous passage if it has fewer than
    ``min_last_words`` words and there is room, and is dropped if it is that
    short and there is no room (e.g. a one-line stub).
    """
    if not 0 < min_last_words <= min_words <= max_words:
        raise ValueError("need 0 < min_last_words <= min_words <= max_words")
    pieces: list[list[str]] = []
    for line in text.split("\n"):
        if len(line.split()) >= min_line_words:
            pieces += _split_long(line, max_words)

    passages: list[list[str]] = []
    current: list[str] = []
    for words in pieces:
        if current and len(current) + len(words) > max_words:
            if len(current) >= min_last_words:     # a very short remainder is dropped
                passages.append(current)
            current = []
        current += words
        if len(current) >= min_words:
            passages.append(current)
            current = []
    if current:
        if len(current) >= min_last_words:
            passages.append(current)
        elif passages and len(passages[-1]) + len(current) <= max_words:
            passages[-1] += current
    return [" ".join(words) for words in passages]


_LATIN = re.compile(r"[A-Za-z]")
_GUJARATI = re.compile(r"[઀-૿]")       # the Gujarati Unicode block


def latin_share(text: str) -> float:
    """Share of Latin letters among the Latin + Gujarati characters of ``text``.

    Some Gujarati Wikipedia articles are mostly untranslated English. Romanized
    queries could match such passages by accident, which is not the effect we
    study, so passages above a threshold are left out of the collection.
    """
    latin = len(_LATIN.findall(text))
    total = latin + len(_GUJARATI.findall(text))
    return latin / total if total else 0.0


def iter_distractors(articles: dict[str, str], skip: Iterable[str], **pack_args) -> Iterator[tuple[str, int, str]]:
    """``(article_id, passage number, text)`` for every passage of every article
    not in ``skip``, in the input order of ``articles``."""
    skip = set(skip)
    for article_id, text in articles.items():
        if article_id in skip:
            continue
        for k, passage in enumerate(pack_passages(clean_wiki_text(text), **pack_args)):
            yield article_id, k, passage
