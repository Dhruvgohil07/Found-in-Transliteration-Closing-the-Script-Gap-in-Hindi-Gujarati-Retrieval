"""Fine-tune the dense retriever on one training configuration (N, 50 or T).

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.train_dense --config configs/train_dense_50.yaml
    python -m scripts.train_dense --config configs/train_dense_50.yaml --time-steps 10 \
        --model intfloat/multilingual-e5-small

A normal run trains for the configured epochs and saves a checkpoint at the end
of each epoch (``output_dir/checkpoint-<step>``, git-ignored), plus the training
log (loss per logging step) both next to the checkpoints and in ``log_file``
(versioned, so the loss curve stays with the results).

``--time-steps N`` is the timing run of task 2: it trains for N steps only, in a
temporary folder that is deleted afterwards, and prints the time per step, the
estimated time for the full run, and the peak GPU memory (on Windows a job that
needs more memory than the card has may spill into system RAM and run many times
slower instead of failing, so the peak is checked against the card's memory).
"""

import argparse
import math
import shutil
import tempfile
import time
from pathlib import Path

import torch

from src.train.train_dense import build_trainer, load_examples, save_log
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--time-steps", type=int, help="timing run: train this many steps only")
    parser.add_argument("--model", help="override the config's model (e.g. e5-small for timing)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    dataset = load_examples(resolve_path(cfg["train_file"]))
    timing = args.time_steps is not None
    output_dir = Path(tempfile.mkdtemp(prefix="train_timing_")) if timing else resolve_path(cfg["output_dir"])
    if not timing and output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"{output_dir} is not empty: remove it or change output_dir")
    trainer = build_trainer(cfg, dataset, output_dir, model_name=args.model, max_steps=args.time_steps)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    card_gb = torch.cuda.get_device_properties(0).total_memory / 2**30

    if timing:
        steps_per_epoch = math.ceil(len(dataset) / cfg["training"]["batch_size"])
        total = steps_per_epoch * cfg["training"]["epochs"]
        per_step = seconds / args.time_steps
        print(f"model {args.model or cfg['model']}: {args.time_steps} steps in {seconds:.0f} s "
              f"({per_step:.1f} s per step, including start-up); full run of ~{total} steps "
              f"({steps_per_epoch} per epoch): ~{per_step * total / 60:.0f} min")
        print(f"peak GPU memory {peak_gb:.2f} GB of {card_gb:.1f} GB"
              + ("  <-- above the card's memory: it would spill into system RAM" if peak_gb > card_gb else ""))
        shutil.rmtree(output_dir, ignore_errors=True)
        return

    save_log(trainer, output_dir / "log_history.json")
    log_file = resolve_path(cfg["log_file"])      # versioned copy: the loss curve of the run
    log_file.parent.mkdir(parents=True, exist_ok=True)
    save_log(trainer, log_file)
    print(f"trained in {seconds / 60:.1f} min, peak GPU memory {peak_gb:.2f} GB -> {output_dir}")


if __name__ == "__main__":
    main()
