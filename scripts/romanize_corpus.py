"""Romanize the passages of a corpus with Uroman (Phase 6, document side).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.romanize_corpus --config configs/docside_romanize_corpus_hi.yaml --sample
    python -m scripts.romanize_corpus --config configs/docside_romanize_corpus_hi.yaml

Each passage (title + text, ``src.data.load.full_text``) is romanized with the
same Uroman call as the rule-based query set of Phase 3
(``romanize_string(text, lang="hin")``), and written as one JSON line
``{"docno", "text"}`` to a git-ignored file that the BM25 index is then built
from (``docs_file`` in an index config).

``--sample`` romanizes only a fixed-seed sample (``sample_fraction`` of the
passages, chosen with ``src.data.splits.split_ids``, so the choice does not
depend on corpus order). It is the timing run of task 3: the script prints the
time per passage and the output size, and from them an estimate for the full
corpus. With ``workers`` > 1 the passages are spread over that many processes
(each loads its own Uroman); the output order stays the corpus order.

An existing output file is never overwritten unless ``--force`` is given.
"""

import argparse
import json
import sys
import time
from collections.abc import Iterator
from multiprocessing import Pool

from src.data.load import full_text, iter_docs
from src.data.splits import split_ids
from src.utils.config import load_config, resolve_path

_romanizer = None  # one Uroman per process (loading its tables takes a few seconds)
_lang = None


def _init_worker(lang: str) -> None:
    global _romanizer, _lang
    import uroman

    _romanizer, _lang = uroman.Uroman(), lang


def _romanize(doc: tuple[str, str]) -> tuple[str, str]:
    docno, text = doc
    return docno, _romanizer.romanize_string(text, lang=_lang)


def sample_docnos(dataset: str, fraction: float, seed: int) -> set[str]:
    """A fixed-seed sample of ``fraction`` of the corpus's docnos."""
    docnos = [doc["docno"] for doc in iter_docs(dataset)]
    sample, _ = split_ids(docnos, n_first=round(fraction * len(docnos)), seed=seed)
    return set(sample)


def romanize_docs(docs: Iterator[tuple[str, str]], lang: str, workers: int) -> Iterator[tuple[str, str]]:
    """Romanize ``(docno, text)`` pairs in order, in ``workers`` processes."""
    if workers == 1:
        _init_worker(lang)
        yield from map(_romanize, docs)
        return
    with Pool(workers, initializer=_init_worker, initargs=(lang,)) as pool:
        yield from pool.imap(_romanize, docs, chunksize=64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample", action="store_true", help="romanize the timing sample only")
    parser.add_argument("--workers", type=int, help="override the config's number of processes")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    workers = args.workers or cfg["workers"]

    out = resolve_path(cfg["sample_file"] if args.sample else cfg["output_file"])
    if out.exists() and not args.force:
        sys.exit(f"output exists, not overwriting: {out} (use --force)")
    out.parent.mkdir(parents=True, exist_ok=True)

    keep = sample_docnos(cfg["dataset"], cfg["sample_fraction"], cfg["seed"]) if args.sample else None
    docs = ((doc["docno"], full_text(doc)) for doc in iter_docs(cfg["dataset"])
            if keep is None or doc["docno"] in keep)

    start = time.perf_counter()
    n = chars = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for docno, text in romanize_docs(docs, cfg["lang"], workers):
            f.write(json.dumps({"docno": docno, "text": text}, ensure_ascii=False) + "\n")
            n += 1
            chars += len(text)
    seconds = time.perf_counter() - start

    size_mb = out.stat().st_size / 2**20
    print(f"romanized {n:,} passages in {seconds / 60:.1f} min with {workers} worker(s) "
          f"({1000 * seconds / n:.1f} ms per passage) -> {out.name}, {size_mb:.1f} MB")
    if args.sample:
        scale = 1 / cfg["sample_fraction"]
        print(f"full-corpus estimate (x{scale:.0f}): {seconds * scale / 60:.0f} min, "
              f"{size_mb * scale / 1024:.2f} GB")


if __name__ == "__main__":
    main()
