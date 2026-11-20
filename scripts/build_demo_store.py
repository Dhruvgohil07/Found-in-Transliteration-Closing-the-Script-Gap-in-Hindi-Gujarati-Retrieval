"""Copy each demo language's passages into a SQLite store (Phase 11).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.build_demo_store --config configs/demo.yaml

For every language in the config, the collection (``dataset``) is streamed and
written to ``store`` (git-ignored, about 0.6 GB for Hindi). The script checks
that the store holds as many passages as the collection.
"""

import argparse
import time

from src.data.load import count_docs, iter_docs
from src.demo.store import PassageStore, build_store
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    for lang, lcfg in cfg["languages"].items():
        start = time.perf_counter()
        n = build_store(iter_docs(lcfg["dataset"]), resolve_path(lcfg["store"]))
        expected = count_docs(lcfg["dataset"])
        if n != expected or len(PassageStore(resolve_path(lcfg["store"]))) != expected:
            raise RuntimeError(f"{lang}: stored {n:,} passages, collection has {expected:,}")
        print(f"{lang}: {n:,} passages -> {lcfg['store']} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
