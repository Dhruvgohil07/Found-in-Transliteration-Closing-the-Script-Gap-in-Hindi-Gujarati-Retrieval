"""Back-transliteration candidates for any query the demo user types (Phase 11).

Words seen in the experiments come from the frozen candidate files
(``data/backtranslit/candidates_{lang}.tsv``), so their candidates are exactly
the ones every reported run used. Any other word is sent to the live IndicXlit
service (``scripts/indicxlit/serve.py`` in Docker), which scores candidates
with the same code. Live scores can differ from a frozen run in the 7th decimal
(float32 arithmetic depends on how words are batched); the candidates and their
order are the same (checked on 50 words per language in Phase 11).

Only the standard library (``urllib``) is used to talk to the service.
"""

import json
import urllib.error
import urllib.request

from src.data.backtranslit import Candidate, Candidates, is_transliterable, tokenize

DEFAULT_URL = "http://127.0.0.1:8765"


class XlitUnavailable(RuntimeError):
    """The live IndicXlit service cannot be reached (Docker not running?)."""


def words_needing_xlit(query: str) -> list[str]:
    """The distinct transliterable words of a romanized query, in query order."""
    seen: list[str] = []
    for token in tokenize(query):
        if is_transliterable(token) and token not in seen:
            seen.append(token)
    return seen


def fetch_candidates(words: list[str], lang: str, url: str = DEFAULT_URL,
                     timeout: float = 30.0) -> Candidates:
    """Ask the live service for the candidates of ``words``."""
    if not words:
        return {}
    body = json.dumps({"lang": lang, "words": words}).encode("utf-8")
    request = urllib.request.Request(f"{url}/candidates", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as e:
        raise XlitUnavailable(f"IndicXlit service error {e.code}: {e.read().decode('utf-8', 'replace')}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise XlitUnavailable(
            f"IndicXlit service not reachable at {url} ({e}). Start it as described in app/README.md."
        ) from e
    return {word: [Candidate(c, p_model, p_word, score) for c, p_model, p_word, score in rows]
            for word, rows in answer.items()}


def candidates_for(query: str, lang: str, frozen: Candidates, url: str = DEFAULT_URL,
                   fetch=fetch_candidates) -> tuple[Candidates, list[str]]:
    """Candidates for every word of ``query``: frozen where known, live otherwise.

    Returns the candidates (only the query's words) and the words that were
    back-transliterated live. ``fetch`` is replaceable for tests.
    """
    words = words_needing_xlit(query)
    known = {w: frozen[w] for w in words if w in frozen}
    new = [w for w in words if w not in frozen]
    live = fetch(new, lang, url) if new else {}
    missing = set(new) - set(live)
    if missing:
        raise XlitUnavailable(f"the service returned no candidates for {sorted(missing)}")
    return {**known, **live}, new
