"""Build the final results table: every method x every query set x every metric (Phase 9).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.final_table --config configs/final_table_hi.yaml

Each row of the config is one method with a run name per query set (``{set}``
is replaced by the set's name; a set the method was not run on is left out).
For every (method, query set) cell the script reads the saved per-query scores
and computes:

- the mean of each metric;
- GapRecovered on ``gap.metric``, against ONE pair of baselines for every row
  (``gap.native`` and ``gap.romanized``), so rows are comparable (SPECS §7.2):
      GapRecovered = (M_method - M_roman) / (M_native - M_roman)
  The native run is restricted to the cell's queries (the hand-romanized set
  covers 50 test queries). Not computed for the native query set;
- a paired t-test on ``test_metric`` against each reference run of that query
  set (e.g. BM25 with romanized queries, and the best single method), with a
  Bonferroni correction over every such comparison in the table.

Every cell is traced to its run file and to the config that produced it
(looked up in results/results.csv). Outputs: a long CSV (one row per cell), a
Markdown table and a LaTeX table of the main metric for the report. All files
are generated; their numbers are never typed by hand.
"""

import argparse
from datetime import date

import pandas as pd

from src.eval.evaluate import RESULTS_FILE, read_per_query
from src.eval.gap import gap_recovered
from src.eval.significance import bonferroni_significant, compare
from src.utils.config import load_config, resolve_path

SET_LABELS = {"native": "native", "roman_natural": "natural", "roman_rule": "rule", "roman_human": "human"}


def run_configs(split: str) -> dict[str, str]:
    """Run name -> config path, from results/results.csv (the last row wins)."""
    log = pd.read_csv(resolve_path(RESULTS_FILE), dtype=str)
    log = log[log["split"] == split]
    names = log["run_file"].str.rsplit("/", n=1).str[-1].str.removesuffix(".trec")
    return dict(zip(names, log["config"], strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    split, metrics, sets = cfg["split"], cfg["metrics"], cfg["query_sets"]
    test_metric, alpha = cfg["test_metric"], cfg["alpha"]
    gap_metric = cfg["gap"]["metric"]
    configs = run_configs(split)

    def per_query(name: str) -> pd.DataFrame:
        return read_per_query(resolve_path(cfg["per_query_dir"]) / f"{name}_{split}.csv")

    def name_for(pattern: str | None, qset: str) -> str | None:
        return None if pattern is None else pattern.format(set=qset) + f"_{split}"

    # 1. One record per (method, query set) cell, with its per-query scores.
    cells = []
    for row in cfg["rows"]:
        for qset in sets:
            pattern = row["runs"].get(qset, row["runs"].get("roman") if qset != "native" else None)
            if pattern is None:
                continue
            name = pattern.format(set=qset)
            cells.append({"method": row["label"], "phase": row["phase"], "query_set": qset,
                          "run": name, "pq": per_query(name)})

    # 2. Every comparison against a reference run (not the reference itself), for Bonferroni.
    def references(qset: str) -> list[tuple[str, str]]:
        return [(ref["label"], ref["runs"][qset]) for ref in cfg["references"] if qset in ref["runs"]]

    n_tests = sum(ref_run != cell["run"] for cell in cells for _, ref_run in references(cell["query_set"]))
    threshold = alpha / n_tests

    records = []
    for cell in cells:
        pq, qset = cell["pq"], cell["query_set"]
        rec = {"method": cell["method"], "phase": cell["phase"], "query_set": qset,
               "n_queries": len(pq), **{m: pq[m].mean() for m in metrics}}
        if qset != "native":
            native = per_query(cfg["gap"]["native"]).loc[pq.index]
            roman = per_query(cfg["gap"]["romanized"].format(set=qset)).loc[pq.index]
            rec["gap_recovered"] = gap_recovered(native[gap_metric].mean(), roman[gap_metric].mean(),
                                                 pq[gap_metric].mean())
        for label, ref_run in references(qset):
            if ref_run == cell["run"]:
                continue
            c = compare(pq, per_query(ref_run).loc[pq.index], test_metric)
            rec[f"diff vs {label}"] = c.diff
            rec[f"p vs {label}"] = c.p
            rec[f"sig vs {label}"] = bonferroni_significant(c.p, n_tests, alpha)
        rec["run_file"] = f"results/runs/{cell['run']}_{split}.trec"
        rec["config"] = configs.get(f"{cell['run']}_{split}", "")
        records.append(rec)
    table = pd.DataFrame(records)

    out_csv = resolve_path(cfg["output"]["csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False, float_format="%.4f", lineterminator="\n")

    # 3. Markdown: the main metric per set (with markers), then every metric per set.
    marks = [(ref["label"], ref["mark"]) for ref in cfg["references"]]

    def marked(rec: pd.Series, m: str) -> str:
        if pd.isna(rec.get(m)):
            return ""
        text = f"{rec[m]:.4f}"
        for label, mark in marks:
            if rec.get(f"sig vs {label}") is True:
                text += mark if rec[f"diff vs {label}"] > 0 else "−" + mark
        return text

    def cell_of(method: str, qset: str) -> pd.Series | None:
        hit = table[(table["method"] == method) & (table["query_set"] == qset)]
        return None if hit.empty else hit.iloc[0]

    legend = "; ".join(f"{mark} = significantly better than {label}, −{mark} = significantly worse"
                       for label, mark in marks)
    lines = [
        f"# {cfg['experiment']}: {cfg['lang']} {split} split",
        "",
        f"Generated by `scripts/final_table.py --config {args.config}` on {date.today().isoformat()}.",
        "",
        f"Paired t-test on {test_metric} against each reference run of the same query set, Bonferroni",
        f"over the table's {n_tests} comparisons (p < {alpha}/{n_tests} = {threshold:.5f}). {legend}.",
        (f"GapRecovered on {gap_metric} against one pair of baselines for every row: "
        f"`{cfg['gap']['native']}` and `{cfg['gap']['romanized']}`."),
        "Each cell's run file and config are in the CSV version of this table.",
        "",
        f"## {gap_metric} (GapRecovered in brackets)",
        "",
        "| Phase | Method | " + " | ".join(f"{SET_LABELS[s]}" for s in sets) + " |",
        "|---|---|" + "---|" * len(sets),
    ]
    for row in cfg["rows"]:
        out = []
        for qset in sets:
            rec = cell_of(row["label"], qset)
            if rec is None:
                out.append("")
            elif qset == "native":
                out.append(marked(rec, gap_metric))
            else:
                out.append(f"{marked(rec, gap_metric)} ({rec['gap_recovered']:.0%})")
        lines.append(f"| {row['phase']} | {row['label']} | " + " | ".join(out) + " |")

    for qset in sets:
        sub = table[table["query_set"] == qset]
        lines += ["", f"## {SET_LABELS[qset]} queries ({int(sub['n_queries'].iloc[0])})", "",
                  "| Phase | Method | " + " | ".join(metrics) + " |",
                  "|---|---|" + "---|" * len(metrics)]
        for _, rec in sub.iterrows():
            lines.append(f"| {rec['phase']} | {rec['method']} | "
                         + " | ".join(marked(rec, m) if m == test_metric else f"{rec[m]:.4f}" for m in metrics)
                         + " |")
    out_md = resolve_path(cfg["output"]["markdown"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # 4. LaTeX (booktabs) of the main metric, for the report.
    tex = ["% Generated by scripts/final_table.py --config " + args.config + "; never edit by hand.",
           "\\begin{tabular}{ll" + "r" * len(sets) + "}", "\\toprule",
           "Phase & Method & " + " & ".join(SET_LABELS[s] for s in sets) + " \\\\", "\\midrule"]
    for row in cfg["rows"]:
        out = []
        for qset in sets:
            rec = cell_of(row["label"], qset)
            if rec is None:
                out.append("--")
                continue
            text = f"{rec[gap_metric]:.3f}"
            for label, mark in marks:
                if rec.get(f"sig vs {label}") is True:
                    sym = cfg["latex_marks"][mark]
                    text += f"$^{{{sym if rec[f'diff vs {label}'] > 0 else '-' + sym}}}$"
            out.append(text)
        tex.append(f"{row['phase']} & {row['label']} & " + " & ".join(out) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    out_tex = resolve_path(cfg["output"]["latex"])
    out_tex.write_text("\n".join(tex) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
