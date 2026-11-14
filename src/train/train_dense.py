"""Fine-tuning a dense retriever on (query, positive, negative) triples (Phase 7).

The model is trained with the InfoNCE loss, SPECS §6:

    L = -log( exp(s(q, d+)/τ) / (exp(s(q, d+)/τ) + Σ_{d- ∈ N(q)} exp(s(q, d-)/τ)) )

where s is the cosine similarity and N(q) holds the example's judged negative
plus every other passage in the batch (in-batch negatives): a batch of 64
examples gives each query 1 positive and 127 negatives.

sentence-transformers pieces used (each explained, since they are not obvious):

- ``CachedMultipleNegativesRankingLoss`` computes this loss for a large batch
  without holding all of it in GPU memory: it first embeds the batch in
  ``mini_batch_size`` chunks without gradients, computes the loss and the
  gradient with respect to each embedding, then re-embeds chunk by chunk with
  gradients ("GradCache", Gao et al. 2021). Same result as a plain batch of 64,
  much less memory. ``scale`` is 1/τ; e5 was trained with τ = 0.01 (Wang et
  al. 2022, §4.1), so scale = 100.
- ``BatchSamplers.NO_DUPLICATES`` never puts the same text twice in a batch. A
  query with two relevant passages has two examples; in one batch, each would
  turn the other's relevant passage into an in-batch negative.
- ``prompts`` adds e5's ``"query: "`` / ``"passage: "`` prefixes per column, as
  at search time (src/retrieval/dense.py).
"""

import json
from pathlib import Path

PROMPTS = {"anchor": "query: ", "positive": "passage: ", "negative": "passage: "}


def load_examples(path: str | Path):
    """A ``datasets.Dataset`` with the columns ``anchor``, ``positive``, ``negative``
    (the loss reads them in this order) from a scripts/build_train_pairs.py file."""
    from datasets import Dataset

    columns = {"anchor": [], "positive": [], "negative": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            columns["anchor"].append(row["query"])
            columns["positive"].append(row["positive"])
            columns["negative"].append(row["negative"])
    return Dataset.from_dict(columns)


def build_trainer(cfg: dict, dataset, output_dir: str | Path, *, model_name: str | None = None,
                  max_steps: int | None = None):
    """A ``SentenceTransformerTrainer`` set up from a train config (configs/train_dense_*.yaml).

    ``model_name`` and ``max_steps`` override the config (used for timing runs).
    """
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    t = cfg["training"]
    model = SentenceTransformer(model_name or cfg["model"])
    model.max_seq_length = cfg["max_seq_length"]
    loss = CachedMultipleNegativesRankingLoss(
        model, scale=cfg["loss"]["scale"], mini_batch_size=cfg["loss"]["mini_batch_size"])
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t["epochs"],
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=t["batch_size"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        fp16=t["fp16"],
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts=PROMPTS,
        save_strategy=t["save_strategy"],
        save_only_model=True,              # no optimizer state: ~1.1 GB per e5-base checkpoint
        logging_steps=t["logging_steps"],
        report_to="none",
        seed=cfg["seed"],
        data_seed=cfg["seed"],
    )
    return SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)


def save_log(trainer, path: str | Path) -> None:
    """Write the trainer's log (loss per logging step, learning rate, epoch) as JSON."""
    Path(path).write_text(json.dumps(trainer.state.log_history, indent=1) + "\n", encoding="utf-8")
