"""Training examples for dense fine-tuning (Phase 7).

One example is a triple ``(query, positive, negative)``:

- **query**: a training query, native (Devanagari) or romanized (the natural
  set of Phase 3), depending on the configuration;
- **positive**: a passage judged relevant to it in the MIRACL qrels;
- **negative**: a passage judged *non-relevant* to it. MIRACL's annotators judged
  passages that a retriever had returned for the query, so these are hard
  negatives (they look relevant), and, unlike negatives taken from a BM25
  ranking, none of them can be an unjudged relevant passage.

There is one example per (query, relevant passage) pair. The three
configurations of the reference paper differ only in the query's script:

- ``N``: every query native;
- ``T``: every query romanized ("transliterated" in the paper);
- ``50``: half of the queries native, half romanized, split by query with a
  fixed seed (``src.data.splits.split_ids``), so a query appears in one script
  only and all three configurations have the same examples and size.

Negatives are drawn per query with ``random.Random(f"{seed}:{qid}")``: the
choice depends only on the seed and the query, never on the order in which
queries are processed. A query's positives get different negatives as long as
it has enough of them.
"""

import random
from collections.abc import Iterable

from src.data.load import Qrels, Queries
from src.data.splits import split_ids

CONFIGS = ("N", "50", "T")


def query_scripts(qids: Iterable[str], config: str, seed: int) -> dict[str, str]:
    """``{qid: "native" | "roman"}`` for a configuration."""
    qids = sorted(set(qids))
    if config == "N":
        return dict.fromkeys(qids, "native")
    if config == "T":
        return dict.fromkeys(qids, "roman")
    if config == "50":
        native, _ = split_ids(qids, n_first=len(qids) // 2, seed=seed)
        native = set(native)
        return {qid: "native" if qid in native else "roman" for qid in qids}
    raise ValueError(f"config must be one of {CONFIGS}, got {config!r}")


def build_examples(native: Queries, roman: Queries, qrels: Qrels, config: str,
                   seed: int) -> list[dict[str, str]]:
    """The training examples of a configuration, with docnos (texts come later).

    Each example: ``qid``, ``script``, ``query``, ``pos_docno``, ``neg_docno``.
    Queries without a judged positive or a judged negative are skipped (in the
    MIRACL-hi training split every query has both). Sorted by qid, then positive.
    """
    if set(native) != set(roman):
        raise ValueError("native and romanized queries must have the same ids")
    scripts = query_scripts(native, config, seed)
    examples = []
    for qid in sorted(native):
        judged = qrels.get(qid, {})
        positives = sorted(d for d, rel in judged.items() if rel > 0)
        negatives = sorted(d for d, rel in judged.items() if rel <= 0)
        if not positives or not negatives:
            continue
        random.Random(f"{seed}:{qid}").shuffle(negatives)
        text = native[qid] if scripts[qid] == "native" else roman[qid]
        for i, pos in enumerate(positives):
            examples.append({"qid": qid, "script": scripts[qid], "query": text,
                             "pos_docno": pos, "neg_docno": negatives[i % len(negatives)]})
    return examples


def attach_texts(examples: list[dict[str, str]], passages: dict[str, str]) -> list[dict[str, str]]:
    """Add ``positive`` and ``negative`` passage texts (``passages``: docno -> text)."""
    missing = sorted({e[k] for e in examples for k in ("pos_docno", "neg_docno")} - set(passages))
    if missing:
        raise KeyError(f"{len(missing)} passages not found, e.g. {missing[0]!r}")
    return [{**e, "positive": passages[e["pos_docno"]], "negative": passages[e["neg_docno"]]}
            for e in examples]
