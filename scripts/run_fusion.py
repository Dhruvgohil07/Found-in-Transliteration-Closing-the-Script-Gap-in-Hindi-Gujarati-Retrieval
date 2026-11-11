"""Fuse the runs of several retrieval configs with RRF, and evaluate the result.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.run_fusion --config configs/docside_dual_rrf.yaml [--split tune]

For each (split, query set) the script reads the run file of every component
config (``results/runs/{lang}_{method}_{query_set}_{split}.trec``, written by
``scripts/run_baseline.py``; it stops with the command to run if one is
missing), fuses them with ``src.retrieval.dual_index.rrf``, writes the fused run,
and logs it like any other run (per-query scores, a row in results/results.csv).

It also prints the fused run next to each component and each ``compare_with``
config on the same queries, with a paired t-test on ``metric``. These printed
comparisons are not logged.
"""

import argparse
import sys
from pathlib import Path

from src.data.load import read_qrels, read_queries
from src.data.splits import subset_qrels
from src.eval.evaluate import RESULTS_FILE, evaluate_run, mean_scores, per_query_scores
from src.eval.significance import compare
from src.retrieval.dual_index import rrf
from src.utils.config import load_config, resolve_path
from src.utils.runs import read_run, run_name, write_run


def component_run_file(config_path: str, query_set: str, split: str) -> tuple[str, Path]:
    """The name and run file of a component config for one query set and split."""
    cfg = load_config(config_path)
    name = run_name(cfg["lang"], cfg["method"], query_set, split)
    return name, resolve_path(Path(cfg["output"]["run_dir"]) / f"{name}.trec")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", help="run only this split (it must be in the config)")
    parser.add_argument("--force", action="store_true", help="redo runs whose file exists")
    args = parser.parse_args()
    cfg = load_config(args.config)
    splits = cfg["splits"]
    if args.split:
        if args.split not in splits:
            raise ValueError(f"split {args.split!r} is not in the config ({splits})")
        splits = [args.split]
    metric = cfg["metric"]

    for split in splits:
        all_qrels = read_qrels(resolve_path(cfg["qrels"].format(split=split)))
        for query_set, pattern in cfg["query_sets"].items():
            queries_path = resolve_path(pattern.format(split=split))
            if not queries_path.exists():
                continue  # e.g. the hand-romanized set exists for test only
            qrels = subset_qrels(all_qrels, read_queries(queries_path))

            parts = {}
            for component in cfg["components"] + cfg.get("compare_with", []):
                name, path = component_run_file(component, query_set, split)
                if not path.exists():
                    sys.exit(f"missing run {path.name}: first run "
                             f"python -m scripts.run_baseline --config {component} --split {split}")
                parts[component] = (name, read_run(path))

            name = run_name(cfg["lang"], cfg["method"], query_set, split)
            run_file = Path(cfg["output"]["run_dir"]) / f"{name}.trec"
            if resolve_path(run_file).exists() and not args.force:
                print(f"{name}: run file exists, skipped (--force to redo)")
                continue
            fused = rrf([parts[c][1] for c in cfg["components"]], k=cfg["rrf_k"], top_k=cfg["top_k"])
            write_run(fused, resolve_path(run_file), tag=name, top_k=cfg["top_k"])
            evaluate_run(
                fused, qrels,
                per_query_file=Path(cfg["output"]["per_query_dir"]) / f"{name}.csv",
                log={"phase": cfg["phase"], "lang": cfg["lang"], "method": cfg["method"],
                     "query_set": query_set, "split": split, "config": args.config,
                     "seed": cfg["seed"], "run_file": run_file.as_posix()},
                results_file=RESULTS_FILE,
            )

            fused_pq = per_query_scores(fused, qrels)
            print(f"\n{name} ({len(qrels)} queries)")
            print(f"  {'run':48} {metric:>8} {'recall@1000':>11}  diff   p (fused vs run)")
            means = mean_scores(fused_pq)
            print(f"  {name:48} {means[metric]:8.4f} {means['recall@1000']:11.4f}")
            for part_name, part_run in parts.values():
                pq = per_query_scores(part_run, qrels)
                c = compare(fused_pq, pq, metric)
                m = mean_scores(pq)
                print(f"  {part_name:48} {m[metric]:8.4f} {m['recall@1000']:11.4f}  {c.diff:+.4f}  {c.p:.3f}")


if __name__ == "__main__":
    main()
