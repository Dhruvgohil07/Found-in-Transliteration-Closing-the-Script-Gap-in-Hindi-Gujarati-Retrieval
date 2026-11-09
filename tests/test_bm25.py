"""Tests for src/retrieval/bm25.py on a tiny toy index (needs Java, like check_env)."""

import pytest

from src.retrieval.bm25 import (
    build_index,
    index_stats,
    negative_idf_terms,
    open_index,
    retrieve,
)

# Toy passages written for these tests (not from any dataset).
DOCS = [
    {"docno": "d1", "text": "राजा राममोहन राय the founder of Brahmo Samaj Roy"},
    {"docno": "d2", "text": "किसान आंदोलन दिल्ली में जारी है"},
]


@pytest.fixture(scope="module")
def toy_index(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("index")
    build_index(DOCS, index_dir)
    index = open_index(index_dir)
    yield index
    index.close()


def terms(index) -> set[str]:
    return {entry.getKey() for entry in index.getLexicon()}


def test_devanagari_and_latin_terms_are_kept_as_written(toy_index):
    lexicon = terms(toy_index)
    assert {"राजा", "राममोहन", "किसान", "दिल्ली"} <= lexicon   # utf tokeniser keeps Devanagari
    assert {"roy", "the", "of"} <= lexicon                    # no stemming, no stopword removal
    assert "roi" not in lexicon                               # (Porter would have made roy -> roi)


def test_index_stats(toy_index):
    stats = index_stats(toy_index)
    assert stats["documents"] == 2
    # d1 has 9 tokens (राजा राममोहन राय the founder of brahmo samaj roy), d2 has 6
    assert stats["tokens"] == 15
    assert stats["avg_doc_length"] == pytest.approx(7.5)


def test_long_docno_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="longer than"):
        build_index([{"docno": "x" * 40, "text": "text"}], tmp_path, docno_length=32)


def test_open_index_without_index_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_index(tmp_path)


# Retrieval needs more passages: with only 2, a term in 1 of them gets
# IDF = log((N - df + 0.5) / (df + 0.5)) = log(1.5 / 1.5) = 0, so every score is 0.
SEARCH_DOCS = DOCS + [
    {"docno": "d3", "text": "भारत ने क्रिकेट मैच जीता"},
    {"docno": "d4", "text": "आज मौसम साफ़ है"},
    {"docno": "d5", "text": "roy roy roy cricket"},
    {"docno": "d6", "text": "दिल्ली की राजधानी"},
    {"docno": "d7", "text": "पुस्तक मेला"},
]


@pytest.fixture(scope="module")
def search_index(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("search_index")
    build_index(SEARCH_DOCS, index_dir)
    index = open_index(index_dir)
    yield index
    index.close()


def test_retrieve_ranks_the_matching_passage_first(search_index):
    run = retrieve(search_index, {"q1": "किसान आंदोलन?", "q2": "Brahmo Samaj founder"},
                   k1=1.2, b=0.75)
    assert max(run["q1"], key=run["q1"].get) == "d2"
    assert max(run["q2"], key=run["q2"].get) == "d1"
    assert all(score > 0 for scores in run.values() for score in scores.values())


def test_retrieve_survives_query_operators(search_index):
    # Without tokenising first, Terrier parses this query as empty and returns nothing.
    run = retrieve(search_index, {"q": "brahmo (samaj) : + - ^ “founder”"}, k1=1.2, b=0.75)
    assert list(run["q"]) == ["d1"]


def test_retrieve_query_without_matches_is_absent(search_index):
    run = retrieve(search_index, {"q": "football"}, k1=1.2, b=0.75)
    assert "q" not in run  # the evaluation then scores this query 0


def test_k1_controls_term_frequency_saturation(search_index):
    # d5 has "roy" 3 times, d1 once. A larger k1 lets repeated occurrences count
    # for more, so d5's lead over d1 grows with k1.
    def lead(k1: float) -> float:
        scores = retrieve(search_index, {"q": "roy"}, k1=k1, b=0.75)["q"]
        return scores["d5"] / scores["d1"]

    assert lead(2.0) > lead(0.5) > 1


def test_negative_idf_terms_are_the_ones_in_over_half_the_passages(search_index):
    # 7 passages: "है" is in d2 and d4 (2 of 7), so no term here passes half.
    assert negative_idf_terms(search_index, {"है", "roy"}) == set()


def test_floor_idf_drops_terms_in_most_passages(tmp_path):
    docs = [{"docno": f"d{i}", "text": f"के word{i}"} for i in range(5)]
    docs.append({"docno": "target", "text": "के किसान"})
    build_index(docs, tmp_path)
    index = open_index(tmp_path)
    try:
        assert negative_idf_terms(index, {"के", "किसान"}) == {"के"}  # in 6 of 6 passages
        classic = retrieve(index, {"q": "के"}, k1=1.2, b=0.75)["q"]
        assert all(score < 0 for score in classic.values())       # negative IDF
        assert retrieve(index, {"q": "के"}, k1=1.2, b=0.75, floor_idf=True) == {}
        floored = retrieve(index, {"q": "किसान के"}, k1=1.2, b=0.75, floor_idf=True)["q"]
        assert list(floored) == ["target"]
    finally:
        index.close()
