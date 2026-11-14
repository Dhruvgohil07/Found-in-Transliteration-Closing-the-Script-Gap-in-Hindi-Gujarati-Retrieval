"""Tests for the Phase 9 analysis helpers (failure buckets, example selection, plots)."""

import pandas as pd

from scripts.failure_buckets import bucket, bucket_order, missed_words, native_words
from scripts.plot_per_query import outcome
from scripts.qualitative_examples import first_by, top_docs

RULES = ["chandrabindu", "nasal", "digits"]


def test_bucket_labels():
    assert [bucket(n, [8, 12]) for n in (1, 8, 9, 12, 13, 30)] == ["1-8", "1-8", "9-12", "9-12", "13+", "13+"]
    assert [bucket(n, [0, 1]) for n in (0, 1, 2, 5)] == ["0", "1", "2+", "2+"]   # edges starting at 0


def test_bucket_order_is_numeric():
    labels = ["13+ words", "9-12 words", "1-8 words", "no entity", "entity"]
    assert sorted(labels, key=bucket_order) == ["1-8 words", "9-12 words", "13+ words", "entity", "no entity"]


def test_native_words_drop_punctuation_and_normalize():
    # the question mark and the danda go; chandrabindu folds to anusvara (Phase 6 key)
    assert native_words("गाँव कहाँ है?", RULES) == native_words("गांव कहां है।", RULES)
    assert len(native_words("गाँव कहाँ है?", RULES)) == 3


def test_missed_words_counts_native_words_absent_from_the_back_transliteration():
    native = "पुर्तगाल से गोवा में लोग आये थे?"
    assert missed_words(native, "पुर्तगाल से गोवा में लोग आये थे", RULES) == 0
    assert missed_words(native, "पोर्तुगाल से जीओए में लोग आये थे", RULES) == 2
    assert missed_words("गाँव", "गांव", RULES) == 0                   # a spelling variant is not a miss


def test_first_by_breaks_ties_by_qid_and_skips():
    s = pd.Series({"q2": 0.5, "q1": 0.5, "q3": -0.2}).rename_axis("qid")
    assert first_by(s, largest=True, skip=set()) == "q1"
    assert first_by(s, largest=True, skip={"q1"}) == "q2"
    assert first_by(s, largest=False, skip=set()) == "q3"


def test_top_docs_and_outcome():
    assert top_docs({"b": 1.0, "a": 1.0, "c": 3.0}, 2) == ["c", "a"]
    a = pd.Series({"q1": 0.5, "q2": 0.2, "q3": 0.0})
    b = pd.Series({"q1": 0.4, "q2": 0.2, "q3": 0.1})
    assert outcome(a, b).to_dict() == {"q1": "win", "q2": "tie", "q3": "loss"}
