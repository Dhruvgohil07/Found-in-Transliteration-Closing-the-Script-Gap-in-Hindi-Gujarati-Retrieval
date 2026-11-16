"""Tests for src/data/distractors.py and the JSON-lines corpus files (Phase 10)."""

import pytest

from src.data.distractors import (
    clean_wiki_text,
    iter_distractors,
    latin_share,
    overlapping_articles,
    pack_passages,
    word_ngrams,
)
from src.data.load import iter_docs, write_docs


def words(n: int, start: int = 0) -> str:
    return " ".join(f"w{i}" for i in range(start, start + n))


# --- cleaning ---------------------------------------------------------------


def test_clean_removes_refs_and_tags_keeps_lines():
    text = 'પૃથ્વી ગ્રહ છે.<ref group="note">નોંધ\nસીએફ. </ref> બીજું<ref name="a"/>\n\n<br>  ત્રીજું   વાક્ય'
    assert clean_wiki_text(text) == "પૃથ્વી ગ્રહ છે. બીજું\nત્રીજું વાક્ય"


# --- overlap filter -----------------------------------------------------------


def test_word_ngrams():
    assert word_ngrams("a b c d", 3) == {"a b c", "b c d"}
    assert word_ngrams("a b", 3) == set()


def test_overlapping_articles_finds_copies_only():
    judged = [words(20)]
    articles = {"copy": "intro " + words(10, start=5) + " outro", "other": words(20, start=100)}
    assert overlapping_articles(articles, judged, n=8) == {"copy": 3}
    assert overlapping_articles(articles, judged, n=8, min_shared=4) == {}


# --- packing ------------------------------------------------------------------


def test_pack_skips_short_lines_and_respects_bounds():
    text = "\n".join(["Heading", words(60), words(60, 60), "a b", words(60, 120), words(60, 180)])
    passages = pack_passages(text, min_words=100, max_words=250)
    assert [len(p.split()) for p in passages] == [120, 120]
    assert "Heading" not in passages[0] and "a b" not in " ".join(passages)


def test_pack_never_exceeds_max_and_splits_long_lines_at_sentences():
    long_line = " ".join(words(40, 40 * i) + "." for i in range(10))   # 400 words, 10 sentences
    passages = pack_passages(long_line, min_words=100, max_words=250)
    assert all(len(p.split()) <= 250 for p in passages)
    assert sum(len(p.split()) for p in passages) == 400
    assert all(p.endswith(".") for p in passages)                       # cut at sentence ends


def test_pack_cuts_one_huge_sentence_into_windows():
    passages = pack_passages(words(600), min_words=100, max_words=250)
    assert [len(p.split()) for p in passages] == [250, 250, 100]


def test_pack_short_leftover_joins_previous_or_is_dropped():
    assert [len(p.split()) for p in pack_passages(words(110) + "\n" + words(10, 200))] == [120]
    assert pack_passages(words(20)) == []                               # one-line stub
    assert [len(p.split()) for p in pack_passages(words(40))] == [40]   # >= min_last_words: kept


def test_pack_rejects_bad_bounds():
    with pytest.raises(ValueError):
        pack_passages("x", min_words=300, max_words=250)


def test_latin_share_counts_letters_only():
    assert latin_share("abc ગુજ") == 0.5
    assert latin_share("ગુજરાત 2023") == 0.0
    assert latin_share("123 ...") == 0.0


def test_iter_distractors_skips_dropped_articles():
    articles = {"1": words(120), "2": words(120, 500), "3": "short"}
    out = list(iter_distractors(articles, skip={"2"}))
    assert [(a, k) for a, k, _ in out] == [("1", 0)]


# --- JSON-lines corpus ----------------------------------------------------------


def test_docs_jsonl_round_trip(tmp_path):
    docs = [{"docno": "indicqa-001", "title": "", "text": "અમદાવાદ શહેર"},
            {"docno": "guwiki-7#0", "title": "", "text": "બીજો ફકરો"}]
    path = tmp_path / "corpus.jsonl"
    assert write_docs(docs, path) == 2
    assert list(iter_docs(str(path))) == docs


def test_write_docs_rejects_duplicate_docno(tmp_path):
    doc = {"docno": "d", "title": "", "text": "x"}
    with pytest.raises(ValueError, match="duplicate docno"):
        write_docs([doc, doc], tmp_path / "c.jsonl")
