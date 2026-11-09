"""List the tune queries where back-transliteration loses most against native.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.backtranslit_failures --config configs/backtranslit_failures_hi.yaml

Queries are ranked by the drop in ``metric`` from the native run to the
back-transliterated run (ties broken by query id); the top ``n`` are written to
``output`` (git-ignored) with the native query, the romanized query, the
back-transliterated query and, word by word, the native words the
back-transliterated query misses and the words it has instead. Only the tune
split is used, so test stays unseen.
"""

import argparse

from src.data.load import read_queries
from src.eval.evaluate import read_per_query
from src.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    if cfg["split"] != "tune":
        raise ValueError("failure analysis uses the tune split only")
    metric = cfg["metric"]

    native_pq = read_per_query(resolve_path(cfg["native_per_query"]))
    method_pq = read_per_query(resolve_path(cfg["method_per_query"]))
    drop = (native_pq[metric] - method_pq.loc[native_pq.index, metric]).rename("drop")
    worst = drop.reset_index().sort_values(["drop", "qid"], ascending=[False, True]).head(cfg["n"])

    native = read_queries(resolve_path(cfg["native_queries"]))
    roman = read_queries(resolve_path(cfg["romanized_queries"]))
    bt = read_queries(resolve_path(cfg["backtranslit_queries"]))

    lines = [f"# Back-transliteration failures ({cfg['split']}, by {metric} drop from native)", ""]
    for row in worst.itertuples(index=False):
        q = row.qid
        native_words = native[q].replace("?", " ").split()
        bt_words = bt[q].split()
        missed = [w for w in native_words if w not in bt_words]
        instead = [w for w in bt_words if w not in native_words]
        lines += [
            (f"## `{q}`: {metric} native {native_pq.loc[q, metric]:.3f} -> "
             f"back-transliterated {method_pq.loc[q, metric]:.3f}"),
            f"- native: {native[q]}",
            f"- romanized: {roman[q]}",
            f"- back-transliterated: {bt[q]}",
            f"- native words missed: {' '.join(missed) or '-'}",
            f"- back-transliterated words instead: {' '.join(instead) or '-'}",
            "",
        ]
    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
