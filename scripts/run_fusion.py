"""Fuse the runs of several retrieval configs (RRF or score interpolation), and evaluate.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.run_fusion --config configs/fusion_A_rrf.yaml [--split tune]

For each (split, query set) the script reads the run file of every component
config (``results/runs/{lang}_{method}_{query_set}_{split}.trec``, written by
``scripts/run_baseline.py``; it stops with the command to run if one is
missing), fuses them with ``src.retrieval.fusion.fuse``, writes the fused run,
and logs it like any other run (per-query scores, a row in results/results.csv).

A component is either a config path, searched with the fused query set itself,
or ``{config: ..., query_set: "{set}_bt1"}`` to take another query set of that
config: ``{set}`` is replaced by the fused query set. This lets a BM25 run that
builds its weighted query from the raw romanized queries (``roman_natural``) be
fused with a dense run on the top-1 back-transliterated ones
(``roman_natural_bt1``); the fused run is named after the romanized set.

The ``fusion`` block names the method and its parameter, chosen on tune by
``scripts/tune_fusion.py``: ``{method: rrf, k: 60}`` or
``{method: interp, alpha: 0.3}`` (alpha weights the first component).

It also prints the fused run next to each component and each ``compare_with``
entry (same form as a component) on the same queries, with a paired t-test on
``metric``. These printed comparisons are not logged.
"""

import argparse
import sys
from pathlib import Path

from src.data.load import read_qrels, read_queries
from src.data.splits import subset_qrels
from src.eval.evaluate import RESULTS_FILE, evaluate_run, mean_scores, per_query_scores
from src.eval.significance import compare
from src.retrieval.fusion import fuse
from src.utils.config import load_config, resolve_path
from src.utils.runs import Run, read_run, run_name, write_run


def component_spec(component: str | dict, query_set: str) -> tuple[str, str]:
    """(config path, query set to read) of a component entry for one fused query set."""
    if isinstance(component, str):
        return component, query_set
    return component["config"], component.get("query_set", "{set}").format(set=query_set)


def component_run_file(config_path: str, query_set: str, split: str) -> tuple[str, Path]:
    """The name and run file of a component config for one query set and split."""
    cfg = load_config(config_path)
    name = run_name(cfg["lang"], cfg["method"], query_set, split)
    return name, resolve_path(Path(cfg["output"]["run_dir"]) / f"{name}.trec")


def load_component_runs(components: list, query_set: str, split: str) -> list[tuple[str, Run]]:
    """(run name, run) of every component for one fused query set and split.

    Stops with the command to run if a run file is missing.
    """
    out = []
    for component in components:
        config_path, component_set = component_spec(component, query_set)
        name, path = component_run_file(config_path, component_set, split)
        if not path.exists():
            sys.exit(f"missing run {path.name}: first run "
                     f"python -m scripts.run_baseline --config {config_path} --split {split}")
        out.append((name, read_run(path)))
    return out


def fusion_params(cfg: dict) -> dict:
    """The ``fusion`` block as keyword arguments of ``fuse``.

    Phase 6's dual-index config predates the block and gives ``rrf_k`` instead.
    """
    if "fusion" in cfg:
        return dict(cfg["fusion"])
    return {"method": "rrf", "k": cfg["rrf_k"]}


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
    params = fusion_params(cfg)

    for split in splits:
        all_qrels = read_qrels(resolve_path(cfg["qrels"].format(split=split)))
        for query_set, pattern in cfg["query_sets"].items():
            queries_path = resolve_path(pattern.format(split=split))
            if not queries_path.exists():
                continue  # e.g. the hand-romanized set exists for test only
            qrels = subset_qrels(all_qrels, read_queries(queries_path))

            name = run_name(cfg["lang"], cfg["method"], query_set, split)
            run_file = Path(cfg["output"]["run_dir"]) / f"{name}.trec"
            if resolve_path(run_file).exists() and not args.force:
                print(f"{name}: run file exists, skipped (--force to redo)")
                continue
            parts = load_component_runs(cfg["components"], query_set, split)
            shown = parts + load_component_runs(cfg.get("compare_with", []), query_set, split)
            fused = fuse([run for _, run in parts], top_k=cfg["top_k"], **params)
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
            for part_name, part_run in shown:
                pq = per_query_scores(part_run, qrels)
                c = compare(fused_pq, pq, metric)
                m = mean_scores(pq)
                print(f"  {part_name:48} {m[metric]:8.4f} {m['recall@1000']:11.4f}  {c.diff:+.4f}  {c.p:.3f}")


if __name__ == "__main__":
    main()
