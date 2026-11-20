"""Tests for src/demo (Phase 11): xlit client, passage store, highlighting and the pipeline.

The pipeline test runs a tiny BM25 index (needs Java, like tests/test_bm25.py)
with configs written to a temporary folder; the live IndicXlit service is
replaced by a fake ``fetch``, so no Docker is needed.
"""

import pytest
import yaml

from src.data.backtranslit import Candidate
from src.demo.pipeline import Pipeline, component_kind, highlight, is_native
from src.demo.store import PassageStore, build_store
from src.demo.xlit_client import (
    XlitUnavailable,
    candidates_for,
    fetch_candidates,
    words_needing_xlit,
)
from src.retrieval.bm25 import build_index


def cands(*texts: str) -> list[Candidate]:
    """Candidates with decreasing scores, in the given order."""
    return [Candidate(t, 0.5 / (i + 1), 0.0, 0.5 / (i + 1)) for i, t in enumerate(texts)]


# --- xlit client --------------------------------------------------------------


def test_words_needing_xlit_dedupes_and_skips_numbers():
    assert words_needing_xlit("Kisan andolan 2020, kisan news!") == ["kisan", "andolan", "news"]


def test_candidates_for_uses_frozen_first_and_fetches_only_new_words():
    frozen = {"kisan": cands("किसान")}
    asked = []

    def fake_fetch(words, lang, url):
        asked.append((tuple(words), lang))
        return {w: cands("नया") for w in words}

    found, live = candidates_for("kisan andolan", "hi", frozen, fetch=fake_fetch)
    assert asked == [(("andolan",), "hi")]
    assert live == ["andolan"]
    assert found["kisan"] == frozen["kisan"] and found["andolan"][0].text == "नया"


def test_candidates_for_fails_if_the_service_skips_a_word():
    with pytest.raises(XlitUnavailable):
        candidates_for("andolan", "hi", {}, fetch=lambda words, lang, url: {})


def test_fetch_candidates_reports_an_unreachable_service():
    with pytest.raises(XlitUnavailable, match="not reachable"):
        fetch_candidates(["kisan"], "hi", url="http://127.0.0.1:9", timeout=2)


# --- helpers --------------------------------------------------------------------


def test_is_native():
    assert is_native("किसान आंदोलन") and is_native("ખેડૂત") and not is_native("kisan andolan")


def test_component_kind():
    assert component_kind({"config": "x", "query_set": "{set}_bt1"}, {}) == "bt1"
    assert component_kind("x", {"backtranslit": {"k": 5}}) == "weighted"
    assert component_kind("x", {}) == "raw"


def test_highlight_marks_normalized_matches_and_keeps_text():
    text = "हिन्दी और आंदोलन, दिल्ली।"
    pieces = highlight(text, {"हिंदी", "आंदोलन"}, rules=["nasal"])
    assert "".join(p for p, _ in pieces) == text
    assert [p for p, matched in pieces if matched] == ["हिन्दी", "आंदोलन"]   # हिन्दी matches via the nasal rule


# --- passage store ----------------------------------------------------------------


def test_store_round_trip(tmp_path):
    docs = [{"docno": "a#1", "title": "शीर्षक", "text": "पहला"}, {"docno": "b#0", "title": "", "text": "दूसरा"}]
    path = tmp_path / "store.sqlite"
    assert build_store(docs, path) == 2
    store = PassageStore(path)
    assert len(store) == 2
    assert store.get(["b#0", "missing"]) == {"b#0": {"title": "", "text": "दूसरा"}}
    assert store.get([]) == {}


def test_store_missing_file_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_demo_store"):
        PassageStore(tmp_path / "none.sqlite")


# --- pipeline on a tiny BM25 index -----------------------------------------------

DOCS = [   # toy passages written for this test
    {"docno": "d1", "text": "किसान आंदोलन दिल्ली में जारी है"},
    {"docno": "d2", "text": "भारत ने क्रिकेट मैच जीता"},
    {"docno": "d3", "text": "आज मौसम साफ़ है"},
    {"docno": "d4", "text": "पुस्तक मेला दिल्ली"},
    {"docno": "d5", "text": "किसान खेती करते हैं"},
    {"docno": "d6", "text": "नदी का पानी"},
]


@pytest.fixture(scope="module")
def demo_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    build_index(DOCS, root / "index")
    (root / "cands.tsv").write_text(
        "word\trank\tcandidate\tp_model\tp_word\tscore\n"
        "kisan\t1\tकिसान\t0.6\t0.5\t0.59\nkisan\t2\tकिसन\t0.4\t0.5\t0.41\n", encoding="utf-8")

    def write(name: str, cfg: dict) -> str:
        path = root / name
        path.write_text(yaml.safe_dump({"experiment": name, "seed": 42, **cfg}, allow_unicode=True),
                        encoding="utf-8")
        return str(path)

    bm25 = {"index_dir": str(root / "index"), "bm25": {"k1": 1.2, "b": 0.75, "floor_idf": False}}
    raw = write("raw.yaml", bm25)
    weighted = write("weighted.yaml", {**bm25, "backtranslit": {"k": 2}})
    fusion = write("fusion.yaml", {"components": [weighted, {"config": raw, "query_set": "{set}_bt1"}],
                                   "fusion": {"method": "interp", "alpha": 0.5}})
    lang_cfg = {
        "candidates_file": str(root / "cands.tsv"),
        "columns": [
            {"title": "raw", "romanized": {"config": raw, "query": "raw"}, "native": {"config": raw, "query": "raw"}},
            {"title": "weighted", "romanized": {"config": weighted, "query": "weighted"},
             "native": {"config": raw, "query": "raw"}},
            {"title": "hybrid", "romanized": {"fusion": fusion}, "native": {"fusion": fusion}},
        ],
    }

    def fake_fetch(words, lang, url):   # the "live service" knows one more word
        return {w: cands("आंदोलन") for w in words if w == "andolan"}

    return Pipeline("hi", lang_cfg, fetch=fake_fetch)


def test_pipeline_romanized_query(demo_pipeline):
    result = demo_pipeline.search("kisan andolan")
    assert not result.native
    assert result.bt1 == "किसान आंदोलन" and result.live_words == ["andolan"]
    raw, weighted, hybrid = result.columns
    assert raw.ranking == []                        # Latin words never match Devanagari passages
    assert weighted.ranking[0][0] == "d1"           # both words back-transliterated: d1 first
    assert {"किसान", "किसन", "आंदोलन"} <= weighted.terms
    assert hybrid.ranking[0][0] == "d1"
    assert result.judged is None


def test_pipeline_native_query_skips_backtransliteration(demo_pipeline):
    result = demo_pipeline.search("किसान खेती")
    assert result.native and result.bt1 == "" and result.live_words == []
    assert result.columns[0].ranking[0][0] == "d5"
