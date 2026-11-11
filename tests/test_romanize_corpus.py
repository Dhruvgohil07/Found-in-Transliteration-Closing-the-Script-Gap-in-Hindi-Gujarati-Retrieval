"""Tests for corpus romanization (scripts/romanize_corpus.py) and for building an
index from a prepared JSON-lines file (scripts/build_bm25_index.py), Phase 6."""

import json

from scripts.build_bm25_index import build_corpus_index, read_docs_file
from scripts.romanize_corpus import romanize_docs
from src.retrieval.bm25 import open_index, read_index_normalization, retrieve

# Toy passages written for these tests (not from any dataset).
DOCS = [
    ("d1", "किसान आंदोलन दिल्ली में जारी है"),
    ("d2", "भारत ने क्रिकेट मैच जीता"),
    ("d3", "आज मौसम साफ़ है"),
    ("d4", "पुस्तक मेला"),
    ("d5", "दिल्ली की राजधानी"),
]


def test_romanize_docs_matches_uroman_and_keeps_order():
    import uroman

    expected = [(docno, uroman.Uroman().romanize_string(text, lang="hin")) for docno, text in DOCS]
    assert list(romanize_docs(iter(DOCS), "hin", workers=1)) == expected
    assert all(not any("ऀ" <= ch <= "ॿ" for ch in text) for _, text in expected)


def test_romanize_docs_with_workers_gives_the_same_output():
    single = list(romanize_docs(iter(DOCS), "hin", workers=1))
    assert list(romanize_docs(iter(DOCS), "hin", workers=2)) == single


def test_index_built_from_a_docs_file(tmp_path):
    docs_file = tmp_path / "docs.jsonl"
    romanized = list(romanize_docs(iter(DOCS), "hin", workers=1))
    docs_file.write_text("".join(json.dumps({"docno": d, "text": t}, ensure_ascii=False) + "\n"
                                 for d, t in romanized), encoding="utf-8")
    assert [d["docno"] for d in read_docs_file(docs_file)] == ["d1", "d2", "d3", "d4", "d5"]

    cfg = {"docs_file": str(docs_file), "index_dir": str(tmp_path / "index"), "tokeniser": "utf",
           "stemmer": "none", "stopwords": "none", "docno_length": 32}
    assert build_corpus_index(cfg) is True
    assert build_corpus_index(cfg) is False                    # exists: not rebuilt
    assert read_index_normalization(tmp_path / "index") == []
    index = open_index(tmp_path / "index")
    try:
        query = dict(romanized)["d1"].split()[0]               # the romanized word for किसान
        run = retrieve(index, {"q": query}, k1=1.2, b=0.75)
        assert max(run["q"], key=run["q"].get) == "d1"
    finally:
        index.close()
