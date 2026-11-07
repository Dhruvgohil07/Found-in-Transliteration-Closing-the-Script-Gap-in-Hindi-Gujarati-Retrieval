"""Tests for src/data/romanize.py. No test calls the Gemini API: a fake client
stands in for it."""

import json
from types import SimpleNamespace

import pytest

from src.data.romanize import (
    GeminiRomanizer,
    cer,
    check_romanized,
    edit_distance,
    normalize_roman,
    parse_batch_response,
    romanize_uroman,
)

# --- rule: Uroman -----------------------------------------------------------

def test_uroman_matches_the_paper_setup():
    # Uroman's known output for these toy queries; no Devanagari may remain.
    out = romanize_uroman({"q1": "किसान आंदोलन", "q2": "भारत की राजधानी क्या है"}, lang="hin")
    assert out == {"q1": "kisaan aamdolan", "q2": "bhaarat kii raajadhaanii kyaa hai"}


# --- checking and parsing LLM output ----------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "kisan आंदोलन", "kisan\tandolan", "kisan\nandolan"])
def test_check_romanized_rejects_unusable_text(bad):
    with pytest.raises(ValueError):
        check_romanized(bad)


def test_parse_batch_response_accepts_a_list_of_the_right_length():
    assert parse_batch_response('["kisan  andolan ", "bharat"]', 2) == ["kisan andolan", "bharat"]


@pytest.mark.parametrize(
    "text",
    ["not json", '{"a": "b"}', '["only one"]', '["a", 3]', '["a", "किसान"]'],
)
def test_parse_batch_response_rejects_wrong_shapes(text):
    with pytest.raises(ValueError):
        parse_batch_response(text, 2)


# --- the Gemini wrapper, with a fake client ---------------------------------

class FakeModels:
    """Answers each call with the next item of ``answers``; an Exception item is raised."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(json.loads(contents))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(text=answer, model_version="fake-model-001")


def make_romanizer(answers):
    client = SimpleNamespace(models=FakeModels(answers))
    waits = []
    romanizer = GeminiRomanizer(
        model="fake-model", instructions="romanize", temperature=0.0, seed=42,
        client=client, max_retries=3, sleep=waits.append,
    )
    return romanizer, client.models, waits


def test_gemini_romanizer_sends_the_batch_as_json_and_records_the_version():
    romanizer, models, waits = make_romanizer(['["kisan andolan", "bharat"]'])
    assert romanizer.romanize_batch(["किसान आंदोलन", "भारत"]) == ["kisan andolan", "bharat"]
    assert models.calls == [["किसान आंदोलन", "भारत"]]
    assert romanizer.model_version == "fake-model-001"
    assert waits == []


def test_gemini_romanizer_retries_errors_and_bad_answers():
    romanizer, models, waits = make_romanizer(
        [RuntimeError("429 rate limit"), '["only one"]', '["kisan", "bharat"]']
    )
    assert romanizer.romanize_batch(["किसान", "भारत"]) == ["kisan", "bharat"]
    assert len(models.calls) == 3
    assert waits == [10, 20]  # the wait doubles after each failure


def test_gemini_romanizer_stops_at_once_on_a_daily_quota_error():
    daily = RuntimeError("429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel")
    romanizer, models, waits = make_romanizer([daily, '["kisan"]'])
    with pytest.raises(RuntimeError, match="daily quota"):
        romanizer.romanize_batch(["किसान"])
    assert len(models.calls) == 1
    assert waits == []


def test_gemini_romanizer_gives_up_after_max_retries():
    romanizer, _, _ = make_romanizer([RuntimeError("down")] * 3)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        romanizer.romanize_batch(["किसान"])


# --- edit distance and CER --------------------------------------------------

@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("kisan", "kisan", 0),
        ("kisaan", "kisan", 1),     # one deletion
        ("kitten", "sitting", 3),   # the textbook example
        ("andolan", "aandolan", 1), # one insertion
    ],
)
def test_edit_distance(a, b, expected):
    assert edit_distance(a, b) == expected
    assert edit_distance(b, a) == expected  # symmetric


def test_normalize_roman_ignores_case_and_spacing():
    assert normalize_roman("  Kisan   ANDOLAN ") == "kisan andolan"


def test_cer_is_total_edits_over_total_reference_length():
    hyps = {"q1": "kisaan", "q2": "Bharat", "extra": "ignored"}
    refs = {"q1": "kisan", "q2": "bharat"}
    # q1: 1 edit over 5 characters; q2: 0 edits (case is ignored) over 6
    assert cer(hyps, refs) == pytest.approx(1 / 11)


def test_cer_needs_common_queries():
    with pytest.raises(ValueError):
        cer({"a": "x"}, {"b": "x"})


# --- hand-romanization sheets (scripts/human_romanization.py) ---------------

def test_human_sheet_round_trip(tmp_path):
    from scripts.human_romanization import read_sheet, write_sheet

    path = tmp_path / "sheet.txt"
    write_sheet({"q2": "भारत", "q1": "किसान आंदोलन"}, path, "a")
    filled = path.read_text(encoding="utf-8").replace("> \n", "> kisan andolan\n", 1)
    path.write_text(filled.replace("> \n", ">  bharat \n", 1), encoding="utf-8")
    assert read_sheet(path) == {"q1": "kisan andolan", "q2": "bharat"}


def test_human_sheet_rejects_a_blank_answer(tmp_path):
    from scripts.human_romanization import read_sheet, write_sheet

    path = tmp_path / "sheet.txt"
    write_sheet({"q1": "किसान"}, path, "b")
    with pytest.raises(ValueError, match="q1"):
        read_sheet(path)
