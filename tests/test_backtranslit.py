"""Tests for src/data/backtranslit.py."""

import pytest

from src.data.backtranslit import (
    Candidate,
    candidate_weights,
    is_transliterable,
    load_candidates,
    load_english,
    tokenize,
    top1_query,
    weighted_query,
    words_to_transliterate,
)


def test_tokenize_lowercases_and_splits_on_punctuation():
    text = "‘Birdie’, rashtrapati dwara anglo-indian 1906 co2 T20 Raja?"
    assert tokenize(text) == [
        "birdie", "rashtrapati", "dwara", "anglo", "indian", "1906", "co2", "t20", "raja",
    ]


def test_only_alphabetic_tokens_are_transliterated():
    assert is_transliterable("kisan")
    assert not is_transliterable("1906")
    assert not is_transliterable("co2")
    assert not is_transliterable("t20")


def test_words_to_transliterate_is_sorted_and_distinct_across_sets():
    natural = {"q1": "kisan andolan", "q2": "Kisan news 2020"}
    rule = {"q1": "kisaan aamdolan"}
    assert words_to_transliterate([natural, rule]) == [
        "aamdolan", "andolan", "kisaan", "kisan", "news",
    ]


# --- weights and queries ---------------------------------------------------------
# Toy candidates in IndicXlit's rank order (text, p_model, p_word, score).
KISAN = [Candidate("किसान", 0.5, 0.9, 0.54), Candidate("किसन", 0.3, 0.1, 0.28),
         Candidate("कीसान", 0.2, 0.0, 0.0)]
RARE = [Candidate("आमगला", 0.6, 0.0, 0.0), Candidate("आम्गला", 0.4, 0.0, 0.0)]
CANDIDATES = {"kisan": KISAN, "andolan": [Candidate("आंदोलन", 1.0, 1.0, 1.0)], "aamgla": RARE}


def test_candidate_weights_renormalise_scores_over_top_k():
    weights = dict(candidate_weights(KISAN, k=2))
    assert weights == pytest.approx({"किसान": 0.54 / 0.82, "किसन": 0.28 / 0.82})
    assert sum(weights.values()) == pytest.approx(1.0)


def test_candidate_with_score_zero_is_dropped():
    # k = 3 includes कीसान, which is not in the frequency dictionary (score 0)
    assert [term for term, _ in candidate_weights(KISAN, k=3)] == ["किसान", "किसन"]


def test_all_zero_scores_fall_back_to_model_probabilities():
    assert dict(candidate_weights(RARE, k=2)) == pytest.approx({"आमगला": 0.6, "आम्गला": 0.4})


def test_top1_query_keeps_numbers_and_unknown_words():
    assert top1_query("Kisan andolan 2020, news?", CANDIDATES) == "किसान आंदोलन 2020 news"


def test_weighted_query_sums_to_one_per_word_and_adds_repeats():
    query = weighted_query("kisan kisan andolan 2020", CANDIDATES, k=2)
    assert query["किसान"] == pytest.approx(2 * 0.54 / 0.82)   # the word appears twice
    assert query["आंदोलन"] == pytest.approx(1.0)
    assert query["2020"] == pytest.approx(1.0)                # not transliterated
    assert sum(query.values()) == pytest.approx(4.0)          # one unit per token


def test_k1_weighted_query_matches_top1(tmp_path):
    path = tmp_path / "candidates.tsv"
    rows = [
        ["word", "rank", "candidate", "p_model", "p_word", "score"],
        ["kisan", "2", "किसन", "0.3", "0.1", "0.28"],     # written out of rank order
        ["kisan", "1", "किसान", "0.5", "0.9", "0.54"],
    ]
    path.write_text("".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
    candidates = load_candidates(path)
    assert [c.text for c in candidates["kisan"]] == ["किसान", "किसन"]  # sorted by rank
    assert weighted_query("kisan", candidates, k=1) == {"किसान": 1.0}
    assert top1_query("kisan", candidates) == "किसान"


# --- English words ---------------------------------------------------------------

FILM = {"film": [Candidate("फिल्म", 0.6, 0.8, 0.62), Candidate("फ़िल्म", 0.4, 0.2, 0.38)]}
ENGLISH = frozenset({"film"})


def test_english_modes_in_top1_query():
    cands = CANDIDATES | FILM
    assert top1_query("kisan film", cands, ENGLISH, "none") == "किसान फिल्म"
    assert top1_query("kisan film", cands, ENGLISH, "replace") == "किसान film"
    assert top1_query("kisan film", cands, ENGLISH, "both") == "किसान फिल्म film"


def test_english_modes_in_weighted_query():
    cands = CANDIDATES | FILM
    replace = weighted_query("film", cands, 2, ENGLISH, "replace")
    assert replace == {"film": 1.0}
    both = weighted_query("film", cands, 2, ENGLISH, "both")
    assert both == pytest.approx({"फिल्म": 0.31, "फ़िल्म": 0.19, "film": 0.5})
    assert sum(both.values()) == pytest.approx(1.0)          # still one unit per word
    # non-English words are unaffected by the mode
    assert weighted_query("kisan", cands, 2, ENGLISH, "both") == weighted_query("kisan", cands, 2)


def test_unknown_english_mode_is_rejected():
    with pytest.raises(ValueError, match="english_mode"):
        top1_query("film", FILM, ENGLISH, "latin")


def test_load_english(tmp_path):
    path = tmp_path / "english.txt"
    path.write_text("".join(f"{w}\n" for w in ["film", "news"]), encoding="utf-8")
    assert load_english(path) == frozenset({"film", "news"})
