"""Tests for src/data/normalize.py (Devanagari and romanized normalization keys, Phase 6).

The Devanagari variant pairs are the ones found in the Phase 5 failure analysis
(docs/plans/05-query-backtranslit.md); the romanized pairs are natural vs Uroman
spellings of the same tune query words. The other words are common Hindi words
chosen to check that a rule does not touch what it should not.
"""

import pandas as pd
import pytest

from src.data.normalize import (
    DEVANAGARI_RULES,
    ROMAN_RULES,
    normalize_devanagari,
    normalize_romanized,
    normalize_weighted,
)

# (as a user types it, as Uroman writes it), from the tune queries
ROMAN_PAIRS = [
    ("ka", "kaa"), ("ki", "kii"), ("tha", "thaa"), ("bharat", "bhaarat"),
    ("mein", "mem"), ("hain", "haim"), ("bandh", "baamdh"), ("deshon", "deshom"),
    ("rashtrapati", "raassttrapati"), ("khiladi", "khilaaddii"),
    ("dwara", "dvaaraa"), ("chunav", "cunaav"),
    ("kitne", "kitane"), ("sabse", "sabase"), ("karta", "karataa"), ("apne", "apane"),
    ("yuddh", "yuddha"), ("dharm", "dharma"),
    ("pehle", "pahale"), ("kehte", "kahate"), ("liye", "lie"), ("gayi", "gaii"),
    ("samvidhan", "samvidhaan"), ("shiksha", "shikssaa"),
]


@pytest.mark.parametrize(("typed", "uroman"), ROMAN_PAIRS)
def test_typed_and_uroman_spellings_share_a_key(typed, uroman):
    assert typed != uroman
    assert normalize_romanized(typed) == normalize_romanized(uroman)


@pytest.mark.parametrize(("rule", "text", "key"), [
    ("ch", "chunav", "cunav"),
    ("sh", "shiksha", "siksa"),
    ("w_v", "dwara", "dvara"),
    ("ph_f", "phool", "fool"),
    ("z_j", "zameen", "jameen"),
    ("anusvara", "mem haim baamdh", "men hain baandh"),
    ("long_vowels", "kaa kii seetaa uumcaa", "ka ki sita umca"),
    ("doubled", "raassttrapati", "rastrapati"),
    ("ei_e", "mein", "men"),
    ("eh_ah", "pehle", "pahle"),
    ("y_glide", "liye gayi", "lie gai"),
    ("schwa", "kitane sabase", "kitne sabse"),
    ("final_a", "yuddha ka", "yuddh k"),
])
def test_each_romanized_rule(rule, text, key):
    assert normalize_romanized(text, [rule]) == key


@pytest.mark.parametrize("word", ["samvidhan", "sampark", "ambar", "kambal"])
def test_anusvara_m_kept_before_labials(word):
    # before p, b, m, v both spellings write m
    assert "m" in normalize_romanized(word, ["anusvara"])


def test_schwa_needs_a_vowel_on_both_sides():
    assert normalize_romanized("bharat", ["schwa"]) == "bharat"        # word end: kept
    assert normalize_romanized("bhaaratiiy", ["long_vowels", "schwa"]) == "bhartiy"
    assert normalize_romanized("kamal", ["schwa"]) == "kamal"          # no vowel after l


def test_romanized_key_lowercases_and_leaves_digits_and_devanagari():
    assert normalize_romanized("Kisan 2021", []) == "kisan 2021"
    assert normalize_romanized("किसान 2021") == "किसान 2021"


def test_char_ngrams_pad_words_and_keep_numbers_whole():
    from src.data.normalize import char_ngrams

    assert char_ngrams("kisan", 3) == "0ki kis isa san an0"
    assert char_ngrams("Kisan", 4) == "0kis kisa isan san0"
    assert char_ngrams("k ka", 3) == "0k0 0ka ka0"          # short words: one padded token
    assert char_ngrams("ka", 4) == "0ka0"
    assert char_ngrams("2016 co2 olympic!", 3) == "2016 co2 0ol oly lym ymp mpi pic ic0"
    with pytest.raises(ValueError):
        char_ngrams("kisan", 1)


def test_char_ngrams_let_near_misses_share_tokens():
    from src.data.normalize import char_ngrams

    a, b = set(char_ngrams("bhartia", 3).split()), set(char_ngrams("bhartiy", 3).split())
    assert a != b and len(a & b) == 5            # 0bh bha har art rti shared; ia0 / iy0 differ


def test_text_key_chains_the_steps_in_order():
    from src.data.normalize import text_key

    assert text_key("आँख") == "आँख"                                    # nothing asked: unchanged
    assert text_key("आँख", rules=["chandrabindu"]) == "आंख"
    assert text_key("kaa raassttrapati", roman_rules=["long_vowels", "doubled"]) == "ka rastrapati"
    assert text_key("kaa", roman_rules=["long_vowels"], ngrams=3) == "0ka ka0"   # key, then n-grams


def test_all_romanized_rules_listed():
    assert ROMAN_RULES == ("ch", "sh", "w_v", "ph_f", "z_j", "anusvara", "long_vowels", "doubled",
                           "ei_e", "eh_ah", "y_glide", "schwa", "final_a")

# (passage / native spelling, back-transliterated spelling) from Phase 5
PHASE5_PAIRS = [
    ("आरम्भिक", "आरंभिक"),
    ("प्रतिबिम्ब", "प्रतिबिंब"),
    ("पम्प", "पंप"),
    ("आँख", "आंख"),
    ("बाँध", "बांध"),
    ("ऊँचा", "ऊंचा"),
    ("सम्राट्", "सम्राट"),
    ("२", "2"),
]


@pytest.mark.parametrize(("native", "backtranslit"), PHASE5_PAIRS)
def test_phase5_variant_pairs_share_a_key(native, backtranslit):
    assert native != backtranslit
    assert normalize_devanagari(native) == normalize_devanagari(backtranslit)


@pytest.mark.parametrize(("word", "key"), [
    ("हिन्दी", "हिंदी"),     # न् before द (dental)
    ("गङ्गा", "गंगा"),       # ङ् before ग (velar)
    ("पञ्च", "पंच"),         # ञ् before च (palatal)
    ("कण्ठ", "कंठ"),         # ण् before ठ (retroflex)
    ("सम्बन्ध", "संबंध"),     # both clusters
])
def test_nasal_of_the_same_class_becomes_anusvara(word, key):
    assert normalize_devanagari(word, ["nasal"]) == key


@pytest.mark.parametrize("word", ["सम्मान", "अन्य", "जन्म", "उन्होंने", "कन्या", "नम्र"])
def test_nasal_before_another_class_is_kept(word):
    # म् before म, न् before य / म / ह, म् before र: no anusvara spelling exists
    assert normalize_devanagari(word) == word


def test_final_virama_only_at_the_end_of_a_word():
    assert normalize_devanagari("किस मुगल सम्राट् ने", ["final_virama"]) == "किस मुगल सम्राट ने"
    assert normalize_devanagari("विद्वान्।", ["final_virama"]) == "विद्वान।"   # danda ends the word
    assert normalize_devanagari("क्या", ["final_virama"]) == "क्या"           # virama inside a word


def test_nukta_both_encodings_and_dropped():
    precomposed = "ज़मीन"         # ज़ as one code point (U+095B)
    decomposed = "ज़मीन"    # ज + nukta
    assert precomposed != decomposed
    assert normalize_devanagari(precomposed, ["nukta"]) == "जमीन"
    assert normalize_devanagari(decomposed, ["nukta"]) == "जमीन"
    assert normalize_devanagari("खिलाड़ी", ["nukta"]) == "खिलाडी"


def test_zero_width_joiner_removed():
    assert normalize_devanagari("क्‍या", ["zero_width"]) == "क्या"
    assert normalize_devanagari("क्‌ष", ["zero_width"]) == "क्ष"


def test_digits():
    assert normalize_devanagari("मंगल के २ उपग्रह, सन् २०२३", ["digits"]) == "मंगल के 2 उपग्रह, सन् 2023"


def test_each_rule_only_does_its_own_change():
    assert normalize_devanagari("आँख पम्प", ["chandrabindu"]) == "आंख पम्प"
    assert normalize_devanagari("आँख पम्प", ["nasal"]) == "आँख पंप"
    assert normalize_devanagari("आँख पम्प", []) == "आँख पम्प"


def test_rule_order_does_not_matter():
    text = "सम्राट् की आँख, पम्प और २ ज़मीन"
    assert normalize_devanagari(text, ["digits", "nasal", "nukta"]) == \
        normalize_devanagari(text, ["nukta", "nasal", "digits"])


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown rules"):
        normalize_devanagari("आँख", ["chandrabindu", "anusvara"])


def test_latin_text_unchanged_and_key_is_idempotent():
    assert normalize_devanagari("kisan andolan 2021 Roy") == "kisan andolan 2021 Roy"
    text = "भारत का सबसे ऊँचा बाँध टिहरी बाँध है। सम्राट् ने २ पम्प खिलाड़ी क्‍या"
    once = normalize_devanagari(text)
    assert normalize_devanagari(once) == once


def test_normalize_weighted_merges_variant_candidates():
    query = {"आंख": 0.6, "आँख": 0.3, "अर्जुन": 1.0, "सम्राट्": 0.5, "सम्राट": 0.5}
    assert normalize_weighted(query) == pytest.approx({"आंख": 0.9, "अर्जुन": 1.0, "सम्राट": 1.0})
    assert normalize_weighted(query, []) == query      # no rules: unchanged
    assert normalize_weighted(query, ["chandrabindu"]) == pytest.approx(
        {"आंख": 0.9, "अर्जुन": 1.0, "सम्राट्": 0.5, "सम्राट": 0.5})


def test_all_rules_listed():
    assert DEVANAGARI_RULES == ("zero_width", "nukta", "chandrabindu", "nasal", "final_virama", "digits")


def test_terrier_tokeniser_keeps_the_marks_the_rules_change():
    """The rules only matter if the index keeps these characters: Terrier's utf
    tokeniser keeps virama, chandrabindu, nukta and Devanagari digits, but splits
    a word at a zero-width joiner (needs Java, like test_bm25)."""
    import pyterrier as pt

    from src.retrieval.bm25 import start_java

    start_java()
    words = ["सम्राट्", "आँख", "खिलाड़ी", "२", "क्‍या"]
    topics = pd.DataFrame({"qid": [str(i) for i in range(len(words))], "query": words})
    tokens = list(pt.rewrite.tokenise("utf")(topics)["query"])
    assert tokens[:4] == words[:4]
    assert tokens[4] == "क् या"
