"""Build the BM25 (Terrier) index of a corpus.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.build_bm25_index --config configs/index_bm25_hi.yaml

Each passage is indexed as title + text (``src.data.load.full_text``). An
existing index is never overwritten unless ``--force`` is given. At the end the
script prints the collection statistics and checks that every passage of the
corpus was indexed.
"""

import argparse
import sys
import time

import ir_datasets

from src.data.load import full_text, iter_docs
from src.retrieval.bm25 import build_index, index_stats, open_index
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing index")
    args = parser.parse_args()
    cfg = load_config(args.config)

    index_dir = resolve_path(cfg["index_dir"])
    if (index_dir / "data.properties").exists() and not args.force:
        sys.exit(f"index exists, not overwriting: {cfg['index_dir']} (use --force)")

    docs = ({"docno": doc["docno"], "text": full_text(doc)} for doc in iter_docs(cfg["dataset"]))
    start = time.perf_counter()
    build_index(
        docs, index_dir,
        tokeniser=cfg["tokeniser"], stemmer=cfg["stemmer"], stopwords=cfg["stopwords"],
        docno_length=cfg["docno_length"],
    )
    minutes = (time.perf_counter() - start) / 60

    index = open_index(index_dir)
    stats = index_stats(index)
    index.close()
    expected = ir_datasets.load(cfg["dataset"]).docs_count()
    print(f"indexed in {minutes:.1f} min -> {cfg['index_dir']}")
    for name, value in stats.items():
        print(f"  {name}: {value:,.1f}" if isinstance(value, float) else f"  {name}: {value:,}")
    if stats["documents"] != expected:
        raise RuntimeError(f"indexed {stats['documents']:,} documents, corpus has {expected:,}")
    print(f"all {expected:,} passages indexed")


if __name__ == "__main__":
    main()
