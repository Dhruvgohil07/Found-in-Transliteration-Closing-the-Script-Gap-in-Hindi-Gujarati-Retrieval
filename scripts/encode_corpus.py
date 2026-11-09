"""Encode every passage of a corpus with a dense model and cache the vectors.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.encode_corpus --config configs/baseline_dense.yaml

Passages (title + text) are encoded with the ``"passage: "`` prefix, cut at
``passage_max_length`` tokens, and saved in shards under ``embeddings_dir``
(git-ignored). Existing shards are skipped, so an interrupted run resumes. At the
end the script checks that every passage of the corpus has a vector.
"""

import argparse
import time

import ir_datasets

from src.data.load import full_text, iter_docs
from src.retrieval.dense import encode_corpus, load_corpus, load_model
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    set_seed(cfg["seed"])

    model = load_model(cfg["model"], fp16=cfg["fp16"])
    docs = ({"docno": d["docno"], "text": full_text(d)} for d in iter_docs(cfg["dataset"]))
    out_dir = resolve_path(cfg["embeddings_dir"])
    start = time.perf_counter()
    n_new = encode_corpus(model, docs, out_dir, shard_size=cfg["shard_size"],
                          max_length=cfg["passage_max_length"], batch_size=cfg["batch_size"])
    print(f"{n_new} shards encoded in {(time.perf_counter() - start) / 60:.1f} min")

    docnos, vectors = load_corpus(out_dir)
    expected = ir_datasets.load(cfg["dataset"]).docs_count()
    if len(docnos) != expected or len(set(docnos)) != len(docnos):
        raise RuntimeError(f"{len(docnos):,} vectors ({len(set(docnos)):,} distinct), corpus has {expected:,}")
    print(f"all {expected:,} passages encoded: {vectors.shape}, {vectors.nbytes / 1e6:,.0f} MB")


if __name__ == "__main__":
    main()
