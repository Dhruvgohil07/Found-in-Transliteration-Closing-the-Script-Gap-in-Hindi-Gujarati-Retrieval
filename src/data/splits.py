"""Fixed-seed query splits and leakage checks (SPECS §5.2)."""

import random
from collections.abc import Iterable
from itertools import combinations

from src.data.load import Qrels, Queries


def split_ids(ids: Iterable[str], n_first: int, seed: int) -> tuple[list[str], list[str]]:
    """Randomly split ids into two groups of ``n_first`` and the rest.

    The ids are sorted before shuffling, so the result depends only on the set
    of ids and the seed, never on input order. A local ``random.Random(seed)``
    is used instead of the global generator, so other code that draws random
    numbers cannot change the split.
    """
    ids = sorted(set(ids))
    if not 0 < n_first < len(ids):
        raise ValueError(f"n_first must be between 1 and {len(ids) - 1}, got {n_first}")
    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:n_first]), sorted(shuffled[n_first:])


def subset_queries(queries: Queries, qids: Iterable[str]) -> Queries:
    return {qid: queries[qid] for qid in qids}


def subset_qrels(qrels: Qrels, qids: Iterable[str]) -> Qrels:
    return {qid: qrels[qid] for qid in qids if qid in qrels}


def normalize_query(text: str) -> str:
    """Lower-case and collapse whitespace, so near-identical query texts compare equal."""
    return " ".join(text.casefold().split())


def drop_overlapping(queries: Queries, protected: Iterable[Queries]) -> tuple[Queries, list[str]]:
    """Remove queries whose id or normalized text appears in any protected split.

    Used to clean the training split against tune/test: the evaluation splits
    stay intact and the training side gives way. Returns the kept queries and
    the sorted ids that were dropped.
    """
    protected_ids: set[str] = set()
    protected_texts: set[str] = set()
    for split in protected:
        protected_ids.update(split)
        protected_texts.update(normalize_query(t) for t in split.values())
    kept: Queries = {}
    dropped: list[str] = []
    for qid, text in queries.items():
        if qid in protected_ids or normalize_query(text) in protected_texts:
            dropped.append(qid)
        else:
            kept[qid] = text
    return kept, sorted(dropped)


def check_no_overlap(splits: dict[str, Queries]) -> None:
    """Raise ``ValueError`` if any two splits share a query id or a query text.

    Text is compared after ``normalize_query``, because the same question can
    appear under two different ids, and that would leak just as much.
    """
    problems = []
    for (name_a, a), (name_b, b) in combinations(splits.items(), 2):
        shared_ids = a.keys() & b.keys()
        if shared_ids:
            problems.append(f"{name_a}/{name_b}: {len(shared_ids)} shared ids, e.g. {min(shared_ids)!r}")
        texts_b = {normalize_query(t) for t in b.values()}
        shared_texts = sorted(qid for qid, t in a.items() if normalize_query(t) in texts_b)
        if shared_texts:
            problems.append(
                f"{name_a}/{name_b}: {len(shared_texts)} shared texts, e.g. {name_a} query {shared_texts[0]!r}"
            )
    if problems:
        raise ValueError("splits overlap: " + "; ".join(problems))
