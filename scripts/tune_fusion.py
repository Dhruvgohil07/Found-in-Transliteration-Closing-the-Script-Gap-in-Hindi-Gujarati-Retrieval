"""Choose the fusion parameter (RRF k, interpolation alpha) of each hybrid on tune.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.tune_fusion --config configs/tune_fusion_hi.yaml

For every combination (a list of component runs, given as in
``scripts/run_fusion.py``), every query set and every grid value of each
method, the component runs are fused with ``src.retrieval.fusion.fuse`` and
scored. Fusion only reads existing run files, so the whole grid takes seconds.

All metrics are saved to ``output``, together with each component alone
(``method = single``) for reference. The chosen value per combination and
method is the best on ``decision.query_set`` by ``decision.metric``, a rule
fixed before the run; ties go to the value listed first in the grid. Only the
tune split is allowed (SPECS §5.2). Rows are not written to
results/results.csv, which holds reported runs.
"""

import argparse

import pandas as pd

from scripts.run_fusion import load_component_runs
from src.data.load import read_qrels, read_queries
from src.data.splits import subset_qrels
from src.eval.evaluate import mean_scores, per_query_scores
from src.eval.significance import compare
from src.retrieval.fusion import fuse
from src.utils.config import load_config, resolve_path

PARAM = {"rrf": "k", "interp": "alpha"}   # the tuned parameter of each method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    split = cfg["split"]
    if split != "tune":
        raise ValueError(f"tuning is only allowed on the tune split, not {split!r}")
    all_qrels = read_qrels(resolve_path(cfg["qrels"].format(split=split)))
    decision_set, decision_metric = cfg["decision"]["query_set"], cfg["decision"]["metric"]

    rows = []
    per_query = {}   # (combination, method, value, query_set) -> per-query scores
    for combo, spec in cfg["combinations"].items():
        for query_set, pattern in cfg["query_sets"].items():
            qrels = subset_qrels(all_qrels, read_queries(resolve_path(pattern.format(split=split))))
            parts = load_component_runs(spec["components"], query_set, split)
            for name, run in parts:
                pq = per_query_scores(run, qrels)
                per_query[(combo, "single", name, query_set)] = pq
                rows.append({"combination": combo, "method": "single", "param": "run",
                             "value": name, "query_set": query_set, **mean_scores(pq)})
            for method, values in cfg["grid"].items():
                for value in values:
                    fused = fuse([run for _, run in parts], method, top_k=cfg["top_k"],
                                 **{PARAM[method]: value})
                    pq = per_query_scores(fused, qrels)
                    per_query[(combo, method, value, query_set)] = pq
                    rows.append({"combination": combo, "method": method, "param": PARAM[method],
                                 "value": value, "query_set": query_set, **mean_scores(pq)})

    table = pd.DataFrame(rows)
    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f", lineterminator="\n")
    print(f"all scores written to {cfg['output']}")

    # The choice: best value on the decision set, per combination and method.
    metric = decision_metric
    for combo, spec in cfg["combinations"].items():
        print(f"\n{combo}: {spec['label']}  ({decision_set}, {metric})")
        here = table[(table["combination"] == combo) & (table["query_set"] == decision_set)]
        singles = here[here["method"] == "single"]
        best_single = singles.loc[singles[metric].idxmax()]
        for _, row in singles.iterrows():
            print(f"  single {row['value']:42} {row[metric]:.4f}  R@1000 {row['recall@1000']:.4f}")
        for method in cfg["grid"]:
            grid = here[here["method"] == method]
            best = grid.loc[grid[metric].idxmax()]   # idxmax keeps the first of tied values
            values = "  ".join(f"{v}:{s:.4f}" for v, s in zip(grid["value"], grid[metric], strict=True))
            c = compare(per_query[(combo, method, best["value"], decision_set)],
                        per_query[(combo, "single", best_single["value"], decision_set)], metric)
            print(f"  {method:6} {PARAM[method]} grid  {values}")
            print(f"  {method:6} chosen {PARAM[method]} = {best['value']}: {best[metric]:.4f} "
                  f"R@1000 {best['recall@1000']:.4f}  vs best single {c.diff:+.4f} (p = {c.p:.3f})")


if __name__ == "__main__":
    main()
