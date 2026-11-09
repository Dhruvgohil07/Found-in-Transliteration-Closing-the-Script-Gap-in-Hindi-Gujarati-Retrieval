"""Back-transliteration of romanized queries (Latin -> Devanagari), Phase 5.

A romanized query is split into tokens; each purely alphabetic token is a word
to back-transliterate. The candidates for every distinct word are produced once
by IndicXlit (in a Linux container, see ``docker/indicxlit/``) and frozen to a
file; queries are then rebuilt from that file on Windows.

Tokenisation: lower-case, then maximal runs of ``[a-z0-9]``. So punctuation and
quotes disappear and ``anglo-indian`` gives ``anglo`` and ``indian``, the same
split Terrier's ``utf`` tokeniser makes. Tokens with digits (``1906``, ``co2``,
``t20``) are kept as they are: IndicXlit transliterates letters, and numbers
appear as digits in the Hindi passages too.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.load import Queries

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-cased ``[a-z0-9]+`` tokens of a romanized query."""
    return TOKEN.findall(text.lower())


def is_transliterable(token: str) -> bool:
    """Only purely alphabetic tokens are back-transliterated."""
    return token.isalpha()


def words_to_transliterate(query_sets: Iterable[Queries]) -> list[str]:
    """Every distinct transliterable word in the given query sets, sorted."""
    words = {
        token
        for queries in query_sets
        for text in queries.values()
        for token in tokenize(text)
        if is_transliterable(token)
    }
    return sorted(words)


# --- candidates and query building ---------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One IndicXlit candidate for a romanized word (a row of the candidates file)."""

    text: str
    p_model: float   # beam probability, normalised over the word's candidates
    p_word: float    # corpus word frequency, normalised over the candidates found in it
    score: float     # IndicXlit's ranking score: 0.9 p_model + 0.1 p_word (0 if not in corpus)


Candidates = dict[str, list[Candidate]]


def load_candidates(path: str | Path) -> Candidates:
    """``{word: [Candidate, ...]}`` in IndicXlit's rank order."""
    table = pd.read_csv(path, sep="\t", dtype={"word": str, "candidate": str},
                        keep_default_na=False).sort_values(["word", "rank"])
    candidates: Candidates = {}
    for row in table.itertuples(index=False):
        candidates.setdefault(row.word, []).append(
            Candidate(row.candidate, row.p_model, row.p_word, row.score))
    return candidates


def candidate_weights(candidates: list[Candidate], k: int) -> list[tuple[str, float]]:
    """The top-``k`` candidates with weights that sum to 1 (the plan's p_{w,j}).

    Weights are IndicXlit's scores renormalised over the top k. If all k scores
    are 0 (no candidate is in IndicXlit's word-frequency dictionary, e.g. for
    rare Uroman spellings), the model probabilities are used instead. A
    candidate with score 0 next to others with a score keeps weight 0 and is
    dropped.
    """
    top = candidates[:k]
    values = [c.score for c in top]
    if sum(values) == 0:
        values = [c.p_model for c in top]
    total = sum(values)
    return [(c.text, v / total) for c, v in zip(top, values, strict=True) if v > 0]


ENGLISH_MODES = ("none", "replace", "both")


def load_english(path: str | Path) -> frozenset[str]:
    """The frozen list of words judged English (scripts/backtranslit_english_words.py)."""
    return frozenset(Path(path).read_text(encoding="utf-8").split())


def _check_mode(english_mode: str) -> None:
    if english_mode not in ENGLISH_MODES:
        raise ValueError(f"english_mode must be one of {ENGLISH_MODES}, got {english_mode!r}")


def _stays_latin(token: str, candidates: Candidates, english: frozenset[str],
                 english_mode: str) -> bool:
    """A token is not back-transliterated if it is not a plain word, has no
    candidates, or is an English word under ``english_mode="replace"``."""
    if not (is_transliterable(token) and candidates.get(token)):
        return True
    return token in english and english_mode == "replace"


def top1_query(text: str, candidates: Candidates, english: frozenset[str] = frozenset(),
               english_mode: str = "none") -> str:
    """Replace every transliterable token by its best candidate; keep the rest.

    Punctuation is dropped (tokenisation), numbers and mixed tokens stay as
    they are, and a word without candidates is kept in Latin script.
    English words (in ``english``) follow ``english_mode``: ``"none"`` treats
    them like any word, ``"replace"`` keeps them in Latin, ``"both"`` writes
    the best candidate followed by the Latin word.
    """
    _check_mode(english_mode)
    out = []
    for token in tokenize(text):
        if _stays_latin(token, candidates, english, english_mode):
            out.append(token)
        elif token in english and english_mode == "both":
            out += [candidates[token][0].text, token]
        else:
            out.append(candidates[token][0].text)
    return " ".join(out)


def weighted_query(text: str, candidates: Candidates, k: int,
                   english: frozenset[str] = frozenset(),
                   english_mode: str = "none") -> dict[str, float]:
    """The weighted query q' = union over the query's words w of
    {(c_{w,j}, p_{w,j})}_{j=1..k}, as ``{term: weight}``.

    Each word contributes total weight 1, spread over its candidates; a token
    that is not transliterated keeps weight 1. A term that appears for several
    words (or several times) gets the sum of its weights, as repeated query
    terms do in BM25. English words follow ``english_mode`` (see ``top1_query``):
    ``"replace"`` gives the Latin word weight 1, ``"both"`` gives it 0.5 and its
    Devanagari candidates the other 0.5.
    """
    _check_mode(english_mode)
    weights: dict[str, float] = {}
    for token in tokenize(text):
        if _stays_latin(token, candidates, english, english_mode):
            pairs = [(token, 1.0)]
        elif token in english and english_mode == "both":
            pairs = [(term, w / 2) for term, w in candidate_weights(candidates[token], k)]
            pairs.append((token, 0.5))
        else:
            pairs = candidate_weights(candidates[token], k)
        for term, weight in pairs:
            weights[term] = weights.get(term, 0.0) + weight
    return weights
