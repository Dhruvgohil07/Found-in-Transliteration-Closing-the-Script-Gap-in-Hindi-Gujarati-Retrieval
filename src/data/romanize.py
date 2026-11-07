"""Romanizing native-script queries, and measuring how close two romanizations are.

Three ways to turn a Devanagari query into Latin script (docs/plans/03):

- **rule** (contrast): Uroman, the rule-based romanizer the reference paper used,
  called exactly as in the paper's ``transliterate.py``:
  ``Uroman().romanize_string(text, lang=...)`` on the raw text. It is deterministic,
  but spells every vowel the same way (``किसान आंदोलन`` -> ``kisaan aamdolan``).
- **natural** (main): an LLM rewrites each query the way a user would type it
  (``kisan andolan``). LLM output can change between calls, so it is produced once
  and frozen to a file (see ``scripts/romanize_queries.py``).
- **human**: typed by hand by two people (``scripts/human_romanization.py``).

The character error rate (CER) compares a romanization with a reference one:

    CER = sum of edit distances / sum of reference lengths

where the edit distance (Levenshtein) is the smallest number of single-character
insertions, deletions and substitutions that turn one string into the other.
"""

import json
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

from src.data.load import Queries

# Any character from the Devanagari Unicode block (U+0900-U+097F).
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# --- rule: Uroman -----------------------------------------------------------

def romanize_uroman(queries: Queries, lang: str) -> Queries:
    """Romanize every query with Uroman; ``lang`` is an ISO 639-3 code (``hin``)."""
    import uroman  # imported here: loading its rule tables takes a few seconds

    romanizer = uroman.Uroman()
    return {qid: romanizer.romanize_string(text, lang=lang) for qid, text in queries.items()}


# --- natural: an LLM ---------------------------------------------------------

def check_romanized(text: str) -> None:
    """Raise ``ValueError`` if a romanized query is unusable: empty, still
    containing Devanagari, or containing a tab/newline (it must fit one TSV field)."""
    if not text.strip():
        raise ValueError("empty romanization")
    if DEVANAGARI.search(text):
        raise ValueError(f"romanization still contains Devanagari: {text!r}")
    if "\t" in text or "\n" in text or "\r" in text:
        raise ValueError(f"romanization contains a tab or newline: {text!r}")


def parse_batch_response(response_text: str, n_expected: int) -> list[str]:
    """Read the model's answer: a JSON list with one romanized string per input query,
    in the same order. Raises ``ValueError`` if the shape or any item is wrong."""
    try:
        items = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response is not JSON: {e}") from e
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ValueError("response is not a JSON list of strings")
    if len(items) != n_expected:
        raise ValueError(f"expected {n_expected} romanizations, got {len(items)}")
    items = [" ".join(item.split()) for item in items]  # tidy stray spaces only
    for item in items:
        check_romanized(item)
    return items


def is_daily_quota_error(error: Exception) -> bool:
    """True for a "requests per day" quota error (HTTP 429): unlike a per-minute
    limit, waiting seconds or minutes cannot fix it, so it is not retried."""
    return "RESOURCE_EXHAUSTED" in str(error) and "PerDay" in str(error)


class GeminiRomanizer:
    """Sends batches of native-script queries to a Gemini model and returns
    one romanized string per query.

    Each request holds a JSON list of queries; the model must answer with a JSON
    list of the same length (enforced with Gemini's structured output,
    ``response_schema=list[str]``). Batching cuts the number of API calls, which
    matters under free-tier rate limits.
    """

    def __init__(
        self,
        model: str,
        instructions: str,
        temperature: float,
        seed: int,
        client: Any = None,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        from google import genai
        from google.genai import types

        # genai.Client() reads the key from the GEMINI_API_KEY environment variable.
        self.client = client if client is not None else genai.Client()
        self.model = model
        self.config = types.GenerateContentConfig(
            system_instruction=instructions,
            temperature=temperature,
            seed=seed,
            response_mime_type="application/json",
            response_schema=list[str],
        )
        self.max_retries = max_retries
        self.sleep = sleep
        self.model_version: str | None = None  # the exact version that answered

    def romanize_batch(self, texts: Sequence[str]) -> list[str]:
        """Romanize a batch, retrying with a growing wait on errors (rate limits,
        overloaded server, or an answer of the wrong shape)."""
        prompt = json.dumps(list(texts), ensure_ascii=False)
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=self.config
                )
                items = parse_batch_response(response.text or "", len(texts))
            # Broad on purpose: network, rate-limit and parse errors are all retried.
            except Exception as e:
                if is_daily_quota_error(e):
                    raise RuntimeError(f"daily quota used up, retrying today cannot help: {e}") from e
                if attempt == self.max_retries:
                    raise RuntimeError(f"batch failed after {attempt} attempts: {e}") from e
                wait = 10 * 2 ** (attempt - 1)
                print(f"  attempt {attempt} failed ({type(e).__name__}: {e}); retry in {wait}s")
                self.sleep(wait)
            else:
                self.model_version = getattr(response, "model_version", None) or self.model
                return items
        raise AssertionError("unreachable")


# --- comparing romanizations -----------------------------------------------

def normalize_roman(text: str) -> str:
    """Lower-case and collapse whitespace before comparing romanizations, so that
    capital letters and double spaces do not count as spelling differences."""
    return " ".join(text.casefold().split())


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance, computed row by row (dynamic programming).

    ``previous[j]`` is the distance between the first ``i - 1`` characters of
    ``a`` and the first ``j`` characters of ``b``; each new row extends ``a`` by
    one character.
    """
    if len(a) < len(b):
        a, b = b, a  # keep the rows as short as possible
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,                       # delete char_a
                current[j - 1] + 1,                    # insert char_b
                previous[j - 1] + (char_a != char_b),  # substitute (free if equal)
            ))
        previous = current
    return previous[-1]


def cer(hypotheses: Queries, references: Queries) -> float:
    """Corpus-level character error rate of ``hypotheses`` against ``references``,
    over the queries both contain. Both sides are normalized first."""
    qids = sorted(set(hypotheses) & set(references))
    if not qids:
        raise ValueError("no query ids in common")
    edits = sum(
        edit_distance(normalize_roman(hypotheses[q]), normalize_roman(references[q])) for q in qids
    )
    length = sum(len(normalize_roman(references[q])) for q in qids)
    return edits / length
