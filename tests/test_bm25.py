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


def test_weighted_with_equal_weights_matches_plain_retrieval(search_index):
    from src.retrieval.bm25 import retrieve_weighted

    plain = retrieve(search_index, {"q": "brahmo samaj founder"}, k1=1.2, b=0.75)
    weighted = retrieve_weighted(search_index, {"q": {"brahmo": 1.0, "samaj": 1.0, "founder": 1.0}},
                                 k1=1.2, b=0.75)
    assert weighted["q"] == pytest.approx(plain["q"])


def test_weights_change_which_term_dominates(search_index):
    from src.retrieval.bm25 import retrieve_weighted

    # d1 matches only "roy" (once); d3 matches only "क्रिकेट" (the word for cricket)
    favour_roy = retrieve_weighted(search_index, {"q": {"roy": 0.9, "क्रिकेट": 0.1}}, k1=1.2, b=0.75)
    favour_cricket = retrieve_weighted(search_index, {"q": {"roy": 0.1, "क्रिकेट": 0.9}}, k1=1.2, b=0.75)
    assert favour_roy["q"]["d1"] > favour_roy["q"]["d3"]
    assert favour_cricket["q"]["d3"] > favour_cricket["q"]["d1"]
    # only the ratio of the weights matters
    scaled = retrieve_weighted(search_index, {"q": {"roy": 9.0, "क्रिकेट": 1.0}}, k1=1.2, b=0.75)
    assert scaled["q"] == pytest.approx(favour_roy["q"])


def test_index_normalization_record(tmp_path):
    from src.retrieval.bm25 import (
        check_normalization,
        read_index_normalization,
        write_index_normalization,
    )

    assert read_index_normalization(tmp_path) == []          # no file: raw text (Phase 4 index)
    check_normalization(tmp_path, [])
    write_index_normalization(tmp_path, ["chandrabindu", "nasal"])
    assert read_index_normalization(tmp_path) == ["chandrabindu", "nasal"]
    check_normalization(tmp_path, ["nasal", "chandrabindu"])  # order does not matter
    with pytest.raises(ValueError, match="built with rules"):
        check_normalization(tmp_path, ["nasal"])
    with pytest.raises(ValueError, match="built with rules"):
        check_normalization(tmp_path, [])

    from src.retrieval.bm25 import read_index_roman_normalization

    assert read_index_roman_normalization(tmp_path) == []      # Devanagari-only record
    write_index_normalization(tmp_path, [], ["long_vowels", "doubled"])
    assert read_index_roman_normalization(tmp_path) == ["long_vowels", "doubled"]
    check_normalization(tmp_path, [], ["doubled", "long_vowels"])
    with pytest.raises(ValueError, match="built with romanized rules"):
        check_normalization(tmp_path, [], ["doubled"])

    write_index_normalization(tmp_path, [], [], ngrams=3)
    check_normalization(tmp_path, [], [], ngrams=3)
    with pytest.raises(ValueError, match="n-grams of size 3"):
        check_normalization(tmp_path, [], [])             # word queries on an n-gram index


def test_ngram_index_matches_near_miss_spellings(tmp_path):
    """Character n-grams let a typed spelling (olympic) find a passage with another
    one (olampik) that shares no whole word with it."""
    from src.data.normalize import char_ngrams

    docs = [{"docno": "target", "text": "olampik khel"}] + [
        {"docno": f"d{i}", "text": text} for i, text in enumerate(
            ["kisan andolan", "cricket match", "mausam saf", "pustak mela", "rajdhani dilli"])]
    words_dir, ngram_dir = tmp_path / "words", tmp_path / "ngrams"
    build_index(docs, words_dir)
    build_index([{**d, "text": char_ngrams(d["text"], 3)} for d in docs], ngram_dir)
    words, ngrams = open_index(words_dir), open_index(ngram_dir)
    try:
        assert retrieve(words, {"q": "olympic"}, k1=1.2, b=0.75) == {}
        run = retrieve(ngrams, {"q": char_ngrams("olympic", 3)}, k1=1.2, b=0.75)
        assert max(run["q"], key=run["q"].get) == "target"
    finally:
        words.close()
        ngrams.close()


def test_normalized_index_matches_spelling_variants(tmp_path):
    """The Phase 6 idea end to end: passages and queries both get the key, so a
    query spelling (आंख, पंप) finds a passage written with the other one (आँख, पम्प)."""
    from src.data.normalize import normalize_devanagari, normalize_weighted
    from src.retrieval.bm25 import retrieve_weighted

    docs = [{"docno": "variant", "text": "अर्जुन की आँख और पम्प"}] + SEARCH_DOCS
    raw_dir, norm_dir = tmp_path / "raw", tmp_path / "norm"
    build_index(docs, raw_dir)
    build_index([{**d, "text": normalize_devanagari(d["text"])} for d in docs], norm_dir)
    raw, norm = open_index(raw_dir), open_index(norm_dir)
    try:
        query = {"q": "आंख पंप"}
        assert retrieve(raw, query, k1=1.2, b=0.75) == {}               # exact strings: no match
        normalized = {"q": normalize_devanagari(query["q"])}
        assert list(retrieve(norm, normalized, k1=1.2, b=0.75)["q"]) == ["variant"]
        weighted = normalize_weighted({"आंख": 0.6, "आँख": 0.4})          # two variants -> one term
        assert list(retrieve_weighted(norm, {"q": weighted}, k1=1.2, b=0.75)["q"]) == ["variant"]
    finally:
        raw.close()
        norm.close()


def test_weighted_terms_are_tokenised_and_merged(search_index):
    from src.retrieval.bm25 import retrieve_weighted

    # "Brahmo-Samaj!" is cut into brahmo + samaj; "BRAHMO" merges with brahmo
    a = retrieve_weighted(search_index, {"q": {"Brahmo-Samaj!": 0.5, "BRAHMO": 0.5}}, k1=1.2, b=0.75)
    b = retrieve_weighted(search_index, {"q": {"brahmo": 1.0, "samaj": 0.5}}, k1=1.2, b=0.75)
    assert a["q"] == pytest.approx(b["q"])
