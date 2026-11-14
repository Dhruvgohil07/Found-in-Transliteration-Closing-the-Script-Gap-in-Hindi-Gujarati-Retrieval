"""Score every fine-tuning checkpoint on the tune queries over a fixed candidate pool.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.select_checkpoint --config configs/select_checkpoint_hi.yaml

Encoding the whole corpus takes ~16.5 min per model, so checkpoints are compared
on a **candidate pool** instead: every passage judged for a tune query (relevant
or not) plus the top ``pool_depth`` passages of each run in ``pool_runs``, put
together into one pool that every tune query searches (passages found for other
queries act as distractors). The pool holds every relevant passage, so its
scores are higher than full-corpus scores; they are only used to compare
checkpoints with each other and with the off-the-shelf model, which is scored
the same way. The pool texts are cached in ``pool_file`` (git-ignored).

For each model (``baseline`` plus every ``checkpoint-*`` folder of each training
config), the pool and the queries of every query set are encoded and ranked by
dot product. All metrics go to ``output``; the best checkpoint per training
config by ``metric`` on ``decision_set`` is printed. Only the tune split is used.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch

from src.data.load import full_text, iter_docs, read_qrels, read_queries
from src.data.splits import subset_qrels
from src.eval.evaluate import mean_scores, per_query_scores
from src.retrieval.dense import PASSAGE_PREFIX, encode, load_model, retrieve
from src.utils.config import load_config, resolve_path
from src.utils.runs import read_run


def build_pool(cfg: dict, qrels: dict) -> dict[str, str]:
    """``{docno: text}`` of the candidate pool, cached in ``pool_file``."""
    path = resolve_path(cfg["pool_file"])
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return {d["docno"]: d["text"] for d in map(json.loads, f)}
    docnos = {docno for judged in qrels.values() for docno in judged}
    for name in cfg["pool_runs"]:
        run = read_run(resolve_path(cfg["run_dir"]) / f"{name}.trec")
        for scores in run.values():
            docnos.update(sorted(scores, key=lambda d: (-scores[d], d))[:cfg["pool_depth"]])
    pool = {d["docno"]: full_text(d) for d in iter_docs(cfg["dataset"]) if d["docno"] in docnos}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps({"docno": k, "text": v}, ensure_ascii=False) + "\n" for k, v in sorted(pool.items()))
    return pool


def checkpoints(model_dir: Path) -> list[tuple[int, Path]]:
    """``(step, folder)`` of every ``checkpoint-<step>`` folder, in training order."""
    found = [(int(m.group(1)), p) for p in model_dir.glob("checkpoint-*")
             if (m := re.fullmatch(r"checkpoint-(\d+)", p.name))]
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    cfg = load_config(parser.parse_args().config)
    if cfg["split"] != "tune":
        raise ValueError(f"checkpoint selection uses the tune split only, not {cfg['split']!r}")

    qrels_all = read_qrels(resolve_path(cfg["qrels"]))
    query_sets = {name: read_queries(resolve_path(p)) for name, p in cfg["query_sets"].items()}
    pool = build_pool(cfg, qrels_all)
    pool_docnos = sorted(pool)
    print(f"candidate pool: {len(pool):,} passages for {len(qrels_all)} tune queries")

    models = [("baseline", 0, 0, cfg["baseline_model"])]
    for name, train_config in cfg["train_configs"].items():
        train_cfg = load_config(train_config)
        for epoch, (step, folder) in enumerate(checkpoints(resolve_path(train_cfg["output_dir"])), start=1):
            models.append((name, epoch, step, str(folder)))

    rows = []
    for name, epoch, step, model_path in models:
        model = load_model(model_path, fp16=True)
        vectors = encode(model, [pool[d] for d in pool_docnos], prefix=PASSAGE_PREFIX,
                         max_length=cfg["passage_max_length"], batch_size=cfg["batch_size"])
        doc_vectors = torch.from_numpy(vectors).cuda().float()
        for qset, queries in query_sets.items():
            run = retrieve(model, queries, doc_vectors, pool_docnos, max_length=cfg["query_max_length"],
                           batch_size=cfg["batch_size"], top_k=cfg["top_k"])
            means = mean_scores(per_query_scores(run, subset_qrels(qrels_all, queries)))
            rows.append({"config": name, "epoch": epoch, "step": step, "query_set": qset, **means})
            print(f"{name:>8} epoch {epoch} (step {step:>3}) {qset:14} {cfg['metric']}={means[cfg['metric']]:.4f}")
        del model, doc_vectors
        torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    out = resolve_path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, float_format="%.4f", lineterminator="\n")

    pivot = table.pivot_table(index=["config", "epoch", "step"], columns="query_set",
                              values=cfg["metric"], sort=False)[list(query_sets)]
    print(f"\n{cfg['metric']} on the tune pool:\n{pivot.round(4).to_string()}")
    decision = table[table["query_set"] == cfg["decision_set"]]
    for name in cfg["train_configs"]:
        rows_c = decision[decision["config"] == name]
        best = rows_c.loc[rows_c[cfg["metric"]].idxmax()]          # ties go to the earlier epoch
        print(f"best {name}: epoch {int(best['epoch'])} (checkpoint-{int(best['step'])}), "
              f"{cfg['decision_set']} {cfg['metric']} {best[cfg['metric']]:.4f}")


if __name__ == "__main__":
    main()
