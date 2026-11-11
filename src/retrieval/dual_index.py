"""Dual-index retrieval: fusing ranked lists from two indexes with RRF (Phase 6).

The two routes to native passages make different mistakes:

- **native index** (Devanagari, normalized) searched with the back-transliterated
  query (Phase 5): fails when IndicXlit picks a wrong word;
- **romanized index** (Uroman, normalized, 4-grams) searched with the raw
  romanized query: needs no transliteration model, fails when the spellings
  share too few n-grams.

Both rank the same passages (same docnos), so their lists can be merged.
**Reciprocal Rank Fusion** (Cormack, Clarke & Büttcher, SIGIR 2009) uses only
ranks, not scores, so BM25 scores from different indexes need no rescaling:

    RRF(d) = sum over runs r containing d of  1 / (k + rank_r(d))

A passage near the top of either list gets a large share; one found by both
gets the sum. ``k`` (60 in the paper) damps the difference between ranks 1 and
2 compared with ranks 50 and 51. Phase 8 tunes it; here it is fixed.

Ranks are computed with the same order as our run files (score descending,
ties broken by docid), so the fusion does not depend on dictionary order.
``ranx.fuse(method="rrf")`` computes the same formula, and the tests check that
both agree. It is not used for the runs because it takes ranks from its own
sort, whose tie order is not documented, and it requires every run to hold the
same queries (Terrier returns no list for a query none of whose terms match).
"""

from src.utils.runs import DEFAULT_TOP_K, Run

DEFAULT_RRF_K = 60


def ranked(scores: dict[str, float]) -> list[str]:
    """Docids by score (highest first), ties broken by docid, as in our run files."""
    return [docid for docid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def rrf(runs: list[Run], k: int = DEFAULT_RRF_K, top_k: int | None = DEFAULT_TOP_K) -> Run:
    """Fuse runs with Reciprocal Rank Fusion; returns ``{qid: {docid: rrf_score}}``.

    A query in any run is in the result; a passage missing from a run simply
    gets nothing from it. Only the ``top_k`` best fused passages per query are
    kept (None: all).
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    fused: Run = {}
    for run in runs:
        for qid, scores in run.items():
            out = fused.setdefault(qid, {})
            for rank, docid in enumerate(ranked(scores), start=1):
                out[docid] = out.get(docid, 0.0) + 1.0 / (k + rank)
    if top_k is not None:
        fused = {qid: dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k])
                 for qid, scores in fused.items()}
    return fused
