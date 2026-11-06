"""Reading and writing TREC run files (SPECS §5.4).

A run file has one line per retrieved document:

    qid Q0 docid rank score run_tag

In memory a run is ``{qid: {docid: score}}``, the same shape that
``ir_measures``, ``ranx`` and ``pytrec_eval`` use.
"""

from pathlib import Path

import ir_measures

Run = dict[str, dict[str, float]]

DEFAULT_TOP_K = 1000  # SPECS §5.4: 1000 results per query (Recall@1000)


def run_name(lang: str, method: str, query_set: str, split: str) -> str:
    """Build the standard run name ``{lang}_{method}_{query_set}_{split}`` (SPECS §8)."""
    return f"{lang}_{method}_{query_set}_{split}"


def write_run(run: Run, path: str | Path, tag: str, top_k: int | None = DEFAULT_TOP_K) -> None:
    """Write a run to a TREC run file, keeping the ``top_k`` best documents per query.

    Documents are ranked by score (highest first). Ties are broken by docid,
    so the same scores always produce the same file. Scores are written at
    full precision, so ``read_run`` gives back exactly the same values.
    ``top_k=None`` keeps every document.
    """
    if not tag or any(c.isspace() for c in tag):
        raise ValueError(f"run tag must be non-empty and contain no whitespace: {tag!r}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(run):
            ranked = sorted(run[qid].items(), key=lambda item: (-item[1], item[0]))
            if top_k is not None:
                ranked = ranked[:top_k]
            # float() turns NumPy/torch scalars into plain floats; !r prints
            # the shortest text that parses back to the identical float.
            f.writelines(
                f"{qid} Q0 {docid} {rank} {float(score)!r} {tag}\n"
                for rank, (docid, score) in enumerate(ranked, start=1)
            )


def read_run(path: str | Path) -> Run:
    """Read a TREC run file into ``{qid: {docid: score}}``.

    The rank and tag columns are dropped: evaluation tools rank by score.
    Raises ``ValueError`` if a document appears twice for the same query.
    """
    run: Run = {}
    for doc in ir_measures.read_trec_run(str(path)):
        scores = run.setdefault(doc.query_id, {})
        if doc.doc_id in scores:
            raise ValueError(f"{path}: duplicate doc {doc.doc_id!r} for query {doc.query_id!r}")
        scores[doc.doc_id] = doc.score
    return run
