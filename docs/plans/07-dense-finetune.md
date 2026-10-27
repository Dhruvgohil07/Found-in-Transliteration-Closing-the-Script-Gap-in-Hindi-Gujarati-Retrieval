# Phase 7 — Dense retriever fine-tuning (GPU)

**Status:** Not started · **Days:** 7–10

## Goal
Teach a multilingual dense retriever to map **romanized queries** close to the **native-script passages** they are relevant to, without losing native-query performance.

## Inputs
- Phase 2 training queries/qrels (strictly separate from tune/test)
- Phase 3 natural romanization of the training queries
- Phase 4 BM25 runs (for hard negatives) and off-the-shelf dense baseline

## Tasks
1. **Build training triples** $(q, d^+, d^-)$:
   - $q$: romanized training query (natural romanization); also include the native version of the same query (mixed training).
   - $d^+$: a relevant native passage from the qrels.
   - $d^-$: **BM25 hard negatives**, i.e. high-ranked passages that are not relevant (skip the top few to reduce false negatives).
2. **Model:** start from the base model chosen in Phase 1 (e.g. `multilingual-e5-small` or `-base`).
3. **Training setup for 8 GB VRAM:**
   - sentence-transformers with `CachedMultipleNegativesRankingLoss` (large effective batch with small memory)
   - fp16, max_len ~256 for passages and ~32 for queries
   - fixed seed; config in `configs/train_dense.yaml`
4. **Loss (InfoNCE with in-batch + hard negatives):**
   $$\mathcal{L} = -\log \frac{e^{s(q,d^+)/\tau}}{e^{s(q,d^+)/\tau} + \sum_{d^- \in \mathcal{N}(q)} e^{s(q,d^-)/\tau}}$$
   where $s(q,d)$ is the cosine similarity, $\tau$ the temperature and $\mathcal{N}(q)$ the in-batch and hard negatives.
5. **Model selection** on the tune split (nDCG@10 on romanized tune queries), with checkpoints every N steps.
6. **Evaluate on test** with all query sets, especially:
   - **native** queries → did performance drop? (catastrophic forgetting check)
   - **human-romanized** queries → does it generalize to real spellings, not just the romanizer it was trained on?
7. Small ablation if time allows: romanized-only vs mixed training; with vs without hard negatives.

## Outputs
- `src/train/build_pairs.py`, `src/train/train_dense.py`
- `configs/train_dense.yaml`
- Checkpoint in `models/` (git-ignored; record the path and how to rebuild it)
- Training log (loss curve) and new rows in `results/results.csv`

## Exit criteria
- Fine-tuned model beats the off-the-shelf dense baseline on romanized test queries, or we have a clear explanation of why it doesn't.
- Native-query performance reported next to it.

## Review gate
- Compare off-the-shelf vs fine-tuned on every query set.
- Check the human-romanized result specifically: real generalization or overfitting to the romanizer?

## Risks / fallback
- Out of memory → smaller model, shorter max_len, smaller mini-batch inside the cached loss.
- Training too slow → train on a subset of the pairs; document the size.
- No gain → still a valid result; analyse why in Phase 9.

## What we found
_(fill in at the end of the phase)_
