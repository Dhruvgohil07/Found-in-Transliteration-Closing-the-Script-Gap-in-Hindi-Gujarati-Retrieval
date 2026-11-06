"""Measure candidate dense models on this GPU, to choose the starting model.

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.bench_dense_models

For each model it reports, from the model itself:
- size: parameters, embedding dimension, maximum input length;
- encoding: passages per second and peak GPU memory (fp16 inference);
- fine-tuning: peak GPU memory of one contrastive training step (fp16
  autocast, AdamW) at several batch sizes, or OOM if it does not fit. A peak
  above the GPU's total memory is flagged: the step only ran by spilling into
  system RAM, which is far too slow for real training.

The input text is placeholder Hindi, truncated to fixed token lengths: only
lengths matter for speed and memory here, not content. These numbers are
about hardware fit, not retrieval quality.
"""

import argparse
import gc
import time

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.utils.seed import set_seed

MODELS = [
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-m3",
]
QUERY_TOKENS = 32
PASSAGE_TOKENS = 256
TRAIN_BATCH_SIZES = [8, 16, 32]

# Long enough to fill PASSAGE_TOKENS after tokenization.
FILLER = "किसान आंदोलन दिल्ली में जारी है और सरकार से बातचीत चल रही है। " * 40


def gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GB"


def reset_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def describe(model: SentenceTransformer) -> str:
    n_params = sum(p.numel() for p in model.parameters())
    return (
        f"params {n_params / 1e6:.0f}M, dim {model.get_embedding_dimension()}, "
        f"max length {model.tokenizer.model_max_length}"
    )


def bench_encoding(model: SentenceTransformer, n_passages: int, batch_size: int) -> str:
    model.half()
    model.max_seq_length = PASSAGE_TOKENS
    passages = [FILLER] * n_passages
    model.encode(passages[:batch_size], batch_size=batch_size)  # warm-up
    torch.cuda.synchronize()
    reset_gpu()
    start = time.perf_counter()
    model.encode(passages, batch_size=batch_size)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return (
        f"{n_passages / seconds:.0f} passages/s at {PASSAGE_TOKENS} tokens, "
        f"peak {gb(torch.cuda.max_memory_allocated())}"
    )


def tokenize_to_gpu(model: SentenceTransformer, n_tokens: int, batch_size: int) -> dict:
    """Tokenize a batch truncated to ``n_tokens`` and move its tensors to the GPU."""
    model.max_seq_length = n_tokens
    features = model.preprocess([FILLER] * batch_size)
    # preprocess() also returns non-tensor entries (e.g. "modality"); keep them as is.
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in features.items()}


def train_step_peak(model_name: str, batch_size: int) -> str:
    """One InfoNCE step with in-batch negatives; returns peak memory or OOM."""
    reset_gpu()
    model = SentenceTransformer(model_name, device="cuda")  # fp32 master weights
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    try:
        q = tokenize_to_gpu(model, QUERY_TOKENS, batch_size)
        p = tokenize_to_gpu(model, PASSAGE_TOKENS, batch_size)
        with torch.autocast("cuda", dtype=torch.float16):
            q_emb = F.normalize(model(q)["sentence_embedding"], dim=-1)
            p_emb = F.normalize(model(p)["sentence_embedding"], dim=-1)
            scores = q_emb @ p_emb.T / 0.05
            loss = F.cross_entropy(scores.float(), torch.arange(batch_size, device="cuda"))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        result = gb(peak)
        # On Windows the driver can spill into system RAM instead of raising
        # OOM: the step "fits" but runs many times slower.
        if peak > torch.cuda.get_device_properties(0).total_memory:
            result += " (more than the GPU has: spilled to system RAM)"
    except torch.OutOfMemoryError:
        result = "OOM"
    finally:
        del model, optimizer
        reset_gpu()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--n-passages", type=int, default=512)
    parser.add_argument("--encode-batch", type=int, default=64)
    args = parser.parse_args()

    set_seed(42)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {gb(props.total_memory)} total\n")

    for name in args.models:
        print(name)
        model = SentenceTransformer(name, device="cuda")
        print(f"  size:      {describe(model)}")
        print(f"  encoding:  {bench_encoding(model, args.n_passages, args.encode_batch)}")
        del model
        reset_gpu()
        steps = ", ".join(
            f"batch {b}: {train_step_peak(name, b)}" for b in TRAIN_BATCH_SIZES
        )
        print(f"  fine-tune: {steps}")
        print()


if __name__ == "__main__":
    main()
