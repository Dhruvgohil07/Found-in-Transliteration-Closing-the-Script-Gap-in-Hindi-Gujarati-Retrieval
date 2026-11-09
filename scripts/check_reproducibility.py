"""Reproducibility check: re-run a baseline from scratch and compare with the stored results.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.check_reproducibility --config configs/baseline_bm25.yaml \\
        --rebuild-index configs/index_bm25_hi.yaml
    python -m scripts.check_reproducibility --config configs/baseline_dense.yaml \\
        --reencode-shards 1

Everything is redone in a temporary folder with the same config and seed;
``results/`` and ``results/results.csv`` are not touched. Then, for every run of
the config:

- the new TREC run file must be **byte-identical** to the stored one, and
- the per-query scores (all metrics) must be identical.

``--rebuild-index`` (BM25) first rebuilds the index from the corpus with the
given index config, so the check covers indexing too. ``--reencode-shards N``
(dense) re-encodes the first N corpus shards and reports how far the new vectors
are from the cached ones (GPU float16 encoding is not guaranteed bit-exact).
The script exits with code 1 if anything differs.
"""

import argparse
import filecmp
import sys
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np

from scripts.run_baseline import run_all
from src.data.load import full_text, iter_docs
from src.eval.evaluate import read_per_query
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def rebuild_index(index_config: str, index_dir: Path) -> None:
    from src.retrieval.bm25 import build_index

    icfg = load_config(index_config)
    docs = ({"docno": d["docno"], "text": full_text(d)} for d in iter_docs(icfg["dataset"]))
    build_index(docs, index_dir, tokeniser=icfg["tokeniser"], stemmer=icfg["stemmer"],
                stopwords=icfg["stopwords"], docno_length=icfg["docno_length"])
    print(f"index rebuilt from scratch in {index_dir}")


def reencode_shards(cfg: dict, n_shards: int) -> bool:
    """Re-encode the first ``n_shards`` shards and compare with the cache; True if
    the same docnos come back and the vectors agree to float16 rounding."""
    from src.retrieval.dense import PASSAGE_PREFIX, encode, load_model

    model = load_model(cfg["model"], fp16=cfg["fp16"])
    cache = resolve_path(cfg["embeddings_dir"])
    docs = iter_docs(cfg["dataset"])
    same = True
    for i in range(n_shards):
        shard = list(islice(docs, cfg["shard_size"]))
        cached_docnos = (cache / f"shard_{i:03d}.txt").read_text(encoding="utf-8").splitlines()
        if [d["docno"] for d in shard] != cached_docnos:
            print(f"shard {i}: DOCNOS DIFFER from the cache")
            same = False
            continue
        new = encode(model, [full_text(d) for d in shard], prefix=PASSAGE_PREFIX,
                     max_length=cfg["passage_max_length"], batch_size=cfg["batch_size"])
        old = np.load(cache / f"shard_{i:03d}.npy")
        diff = np.abs(new.astype(np.float32) - old.astype(np.float32))
        identical = int((diff == 0).all(axis=1).sum())
        print(f"shard {i}: {identical:,} of {len(old):,} vectors bit-identical; "
              f"max |difference| {diff.max():.2e}, mean {diff.mean():.2e}")
        same &= bool(diff.max() <= 1e-3)  # float16 has ~3 significant digits
    return same


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="a baseline config")
    parser.add_argument("--rebuild-index", metavar="INDEX_CONFIG",
                        help="BM25: rebuild the index from scratch with this index config")
    parser.add_argument("--reencode-shards", type=int, default=0,
                        help="dense: re-encode this many corpus shards and compare")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    stored = cfg["output"]
    ok = True

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        if args.rebuild_index:
            rebuild_index(args.rebuild_index, tmp / "index")
            cfg["index_dir"] = str(tmp / "index")
        if args.reencode_shards:
            ok &= reencode_shards(cfg, args.reencode_shards)

        cfg["output"] = {"run_dir": str(tmp / "runs"), "per_query_dir": str(tmp / "per_query")}
        means = run_all(cfg, args.config, cfg["splits"], force=True,
                        results_file=tmp / "results.csv")

        print("\n| run | run file | per-query scores |\n|---|---|---|")
        for name in means:
            old_run = resolve_path(stored["run_dir"]) / f"{name}.trec"
            same_run = old_run.exists() and filecmp.cmp(old_run, tmp / "runs" / f"{name}.trec",
                                                        shallow=False)
            old_pq = read_per_query(resolve_path(stored["per_query_dir"]) / f"{name}.csv")
            new_pq = read_per_query(tmp / "per_query" / f"{name}.csv")
            same_pq = old_pq.equals(new_pq)
            print(f"| {name} | {'identical' if same_run else 'DIFFERS'} | "
                  f"{'identical' if same_pq else 'DIFFER'} |")
            ok &= same_run and same_pq

    print(f"\nreproducibility check {'PASSED' if ok else 'FAILED'} for {args.config}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
