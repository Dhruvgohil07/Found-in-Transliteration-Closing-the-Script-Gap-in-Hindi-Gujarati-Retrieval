"""Fusing ranked lists from several retrievers (Phase 6 dual index, Phase 8 hybrids).

Two retrievers that make different mistakes can be combined after retrieval:
each produces its own ranked list for a query, and the lists are merged into
one. Every run here ranks the same MIRACL-hi passages (same docnos), whether it
comes from a BM25 index (native or romanized) or from a dense model.

Two fusion methods:

**Reciprocal Rank Fusion** (Cormack, Clarke & Büttcher, SIGIR 2009) uses only
ranks, not scores:

    RRF(d) = sum over runs r containing d of  1 / (k + rank_r(d))

A passage near the top of either list gets a large share; one found by both
gets the sum. ``k`` (60 in the paper) damps the difference between ranks 1 and
2 compared with ranks 50 and 51: a small ``k`` trusts the very top of each list
more. Since only ranks count, BM25 scores (0 to ~50) and cosine similarities
(~0.7 to 0.9) need no rescaling.

**Score interpolation** (a weighted CombSUM) uses the scores themselves, after
putting them on one scale with min-max normalization per query:

    s_hat_r(d) = (s_r(d) - min_r) / (max_r - min_r)        (min/max over r's list)
    s(d)       = sum over runs r of  w_r * s_hat_r(d)

For two runs with weights (alpha, 1 - alpha) this is

    s(d) = alpha * s_hat_lex(d) + (1 - alpha) * s_hat_dense(d).

Unlike RRF it keeps *how much* better one passage scores than the next. A
passage missing from a run gets 0 from it, the same as the last passage of
that run's list (our runs are 1000 deep, so "missing" means "below rank 1000").

Ranks are computed with the same order as our run files (score descending,
ties broken by docid), so the fusion does not depend on dictionary order.
``ranx.fuse`` (``method="rrf"``, and ``norm="min-max", method="wsum"``)
computes the same formulas, and the tests check that both agree. It is not used
for the runs because it takes ranks from its own sort, whose tie order is not
documented, and it requires every run to hold the same queries (Terrier returns
no list for a query none of whose terms match). One deliberate difference: a
list whose scores are all equal (e.g. a single passage) is normalized to 1 here
(each passage is the best of its list), while ranx gives 0.
"""

from collections.abc import Sequence

from src.utils.runs import DEFAULT_TOP_K, Run

DEFAULT_RRF_K = 60


def ranked(scores: dict[str, float]) -> list[str]:
    """Docids by score (highest first), ties broken by docid, as in our run files."""
    return [docid for docid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def _keep_top(fused: Run, top_k: int | None) -> Run:
    """Keep the ``top_k`` best passages per query (None: all), in run-file order."""
    if top_k is None:
        return fused
    return {qid: dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k])
            for qid, scores in fused.items()}


def rrf(runs: Sequence[Run], k: int = DEFAULT_RRF_K, top_k: int | None = DEFAULT_TOP_K) -> Run:
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
    return _keep_top(fused, top_k)


def min_max(scores: dict[str, float]) -> dict[str, float]:
    """Rescale one query's scores to [0, 1]: best passage 1, worst 0.

    If every score is equal (e.g. a single passage), every passage gets 1.
    """
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return dict.fromkeys(scores, 1.0)
    return {docid: (s - lo) / (hi - lo) for docid, s in scores.items()}


def interpolate(runs: Sequence[Run], weights: Sequence[float],
                top_k: int | None = DEFAULT_TOP_K) -> Run:
    """Weighted sum of min-max normalized scores; returns ``{qid: {docid: score}}``.

    ``weights[i]`` belongs to ``runs[i]``; they must be non-negative and sum to 1
    (for two runs: ``[alpha, 1 - alpha]``). A passage missing from a run gets 0
    from it. A query in any run is in the result.
    """
    if len(weights) != len(runs):
        raise ValueError(f"{len(runs)} runs but {len(weights)} weights")
    if any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"weights must be non-negative and sum to 1, got {list(weights)}")
    fused: Run = {}
    for run, weight in zip(runs, weights, strict=True):
        for qid, scores in run.items():
            out = fused.setdefault(qid, {})
            for docid, s in min_max(scores).items():
                out[docid] = out.get(docid, 0.0) + weight * s
    return _keep_top(fused, top_k)


def fuse(runs: Sequence[Run], method: str, top_k: int | None = DEFAULT_TOP_K, **params) -> Run:
    """Fuse with ``method="rrf"`` (param ``k``) or ``method="interp"`` (param ``weights``,
    or ``alpha`` for two runs, meaning weights ``[alpha, 1 - alpha]``).

    The single entry point used by the scripts, so a config can name the method.
    """
    if method == "rrf":
        return rrf(runs, k=params.get("k", DEFAULT_RRF_K), top_k=top_k)
    if method == "interp":
        if "alpha" in params:
            if len(runs) != 2:
                raise ValueError("alpha needs exactly two runs; give weights instead")
            weights = [params["alpha"], 1.0 - params["alpha"]]
        else:
            weights = params["weights"]
        return interpolate(runs, weights, top_k=top_k)
    raise ValueError(f"unknown fusion method {method!r} (rrf or interp)")
