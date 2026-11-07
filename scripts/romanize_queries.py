"""Write the romanized query files for one method (rule or natural).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.romanize_queries --config configs/romanize_hi.yaml --method rule
    python -m scripts.romanize_queries --config configs/romanize_hi.yaml --method natural

For each split, ``{split}_queries.tsv`` (native, from Phase 2) becomes
``{split}_queries_roman_{method}.tsv`` in ``output_dir``, with the same query ids.

The files are **frozen**: if one already exists, the script stops instead of
overwriting it (``--force`` overrides this; only for a deliberate re-run).

``natural`` calls the Gemini API. The key is read from ``GEMINI_API_KEY``, either
already set in the environment or written in the git-ignored ``.env`` file at the
repo root (see ``.env.example``).
Finished batches are appended to a progress file, so an interrupted run (rate
limit, network) continues where it stopped instead of paying for the same
queries again. The model id, the exact model version that answered, and the
prompt are saved next to the output in ``natural_meta_{lang}.json``.
"""

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from src.data.load import Queries, read_queries, write_queries
from src.data.romanize import GeminiRomanizer, check_romanized, romanize_uroman
from src.utils.config import load_config, load_env_file, resolve_path


def output_path(cfg: dict, split: str, method: str) -> Path:
    return resolve_path(cfg["output_dir"]) / f"{split}_queries_roman_{method}.tsv"


def romanize_rule(cfg: dict, native: dict[str, Queries]) -> dict[str, Queries]:
    return {split: romanize_uroman(queries, cfg["rule"]["lang"]) for split, queries in native.items()}


def romanize_natural(cfg: dict, native: dict[str, Queries]) -> dict[str, Queries]:
    ncfg = cfg["natural"]
    if not ncfg.get("model"):
        sys.exit("natural.model is not set in the config: choose the Gemini model first")
    load_env_file()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set: put it in .env (see .env.example)")

    # Progress file: one JSON line per finished query (git-ignored working file).
    progress_path = resolve_path(cfg["input_dir"]) / f"natural_progress_{cfg['lang']}.jsonl"
    done: dict[str, dict] = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["qid"]] = row
        print(f"resuming: {len(done)} queries already romanized in {progress_path.name}")

    todo = [
        (qid, text) for queries in native.values() for qid, text in sorted(queries.items())
        if qid not in done
    ]
    romanizer = GeminiRomanizer(
        model=ncfg["model"], instructions=ncfg["instructions"],
        temperature=ncfg["temperature"], seed=cfg["seed"],
    )
    size = ncfg["batch_size"]
    n_batches = (len(todo) + size - 1) // size
    with open(progress_path, "a", encoding="utf-8", newline="\n") as progress:
        for b, start in enumerate(range(0, len(todo), size), start=1):
            batch = todo[start:start + size]
            romanized = romanizer.romanize_batch([text for _, text in batch])
            for (qid, text), roman in zip(batch, romanized, strict=True):
                row = {"qid": qid, "native": text, "roman": roman,
                       "model_version": romanizer.model_version, "batch_size": size}
                progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[qid] = row
            progress.flush()
            print(f"batch {b}/{n_batches} done")
            if b < n_batches:
                time.sleep(ncfg["seconds_between_calls"])

    meta = {
        "model": ncfg["model"],
        "model_versions": sorted({row["model_version"] for row in done.values()}),
        "tool": ncfg["tool"],
        "temperature": ncfg["temperature"],
        "seed": cfg["seed"],
        # The batch size can change between resumed runs (e.g. to fit a daily
        # quota), so every size used is recorded, with its number of queries.
        "queries_per_batch_size": {
            str(b): sum(row["batch_size"] == b for row in done.values())
            for b in sorted({row["batch_size"] for row in done.values()})
        },
        "instructions": ncfg["instructions"],
        "finished_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
    }
    meta_path = resolve_path(cfg["output_dir"]) / f"natural_meta_{cfg['lang']}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        split: {qid: done[qid]["roman"] for qid in queries} for split, queries in native.items()
    }


METHODS = {"rule": romanize_rule, "natural": romanize_natural}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--force", action="store_true", help="overwrite frozen files")
    args = parser.parse_args()
    cfg = load_config(args.config)

    existing = [
        output_path(cfg, split, args.method) for split in cfg["splits"]
        if output_path(cfg, split, args.method).exists()
    ]
    if existing and not args.force:
        sys.exit(f"frozen file exists, not overwriting: {existing[0].name} (use --force)")

    native = {
        split: read_queries(resolve_path(cfg["input_dir"]) / f"{split}_queries.tsv")
        for split in cfg["splits"]
    }
    romanized = METHODS[args.method](cfg, native)

    for split, queries in romanized.items():
        if set(queries) != set(native[split]):
            raise ValueError(f"{split}: romanized query ids differ from the native file")
        for qid, text in queries.items():
            try:
                check_romanized(text)
            except ValueError as e:
                raise ValueError(f"{split} {qid}: {e}") from e
        write_queries(queries, output_path(cfg, split, args.method))
        print(f"{split}: {len(queries)} queries -> {output_path(cfg, split, args.method).name}")


if __name__ == "__main__":
    main()
