"""Write top-1 back-transliterated versions of the romanized query sets.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.build_backtranslit_queries --config configs/backtranslit_queries_hi.yaml

Each romanized query is rebuilt with every word replaced by its best IndicXlit
candidate (``src.data.backtranslit.top1_query``). The files keep the query ids
of the romanized files and are derived from the frozen candidates, so they can
be rebuilt at any time. The script prints how many words had candidates.
"""

import argparse

from src.data.backtranslit import (
    is_transliterable,
    load_candidates,
    tokenize,
    top1_query,
)
from src.data.load import read_queries, write_queries
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    candidates = load_candidates(resolve_path(cfg["candidates_file"]))

    for split in cfg["splits"]:
        for query_set, pattern in cfg["query_sets"].items():
            path = resolve_path(pattern.format(split=split))
            if not path.exists():
                continue
            queries = read_queries(path)
            words = [t for text in queries.values() for t in tokenize(text) if is_transliterable(t)]
            missing = sorted({w for w in words if w not in candidates})
            if missing:
                raise ValueError(f"{split} {query_set}: no candidates for {missing[:5]} "
                                 "(regenerate the candidates file)")
            out = resolve_path(cfg["output"].format(split=split, query_set=query_set))
            write_queries({qid: top1_query(text, candidates) for qid, text in queries.items()}, out)
            print(f"{split} {query_set}: {len(queries)} queries, {len(words):,} words "
                  f"back-transliterated -> {out.name}")


if __name__ == "__main__":
    main()
