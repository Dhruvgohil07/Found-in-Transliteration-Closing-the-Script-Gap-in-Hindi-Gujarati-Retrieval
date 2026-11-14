# Phase 7 — Dense retriever fine-tuning (GPU)

**Status:** Done (2026-11-14), review gate approved · **Days:** 7–10

> **Plan changes (2026-11-12, before the phase started):**
> - **Hard negatives:** MIRACL's *judged* non-relevant passages (9,191 for the 1,168 training
>   queries, at least one per query) replace BM25 negatives. A person judged them non-relevant, so
>   there are no hidden positives, which the "skip the top few" rule for BM25 negatives only
>   reduces; and they are hard, since they were judged after being retrieved. BM25 negatives
>   become the optional ablation.
> - **Configurations:** all three of the paper's set-ups are trained, **N** (native queries),
>   **50** (half native, half romanized) and **T** (romanized), so the pattern of Table 1 can be
>   compared directly (open decision in docs/progress.md, now settled).
> - **Query length:** 64 tokens at search time instead of 32, as the Phase 4 dense baseline
>   (measured over every query set: the longest is 56 tokens with the prefix; 48, first
>   planned, would cut 4 rule test queries). Training does not cut queries.
> - **Timing first** (as for the corpus in Phase 6) and **checkpoint selection on a fixed
>   candidate pool** of tune passages, since re-encoding the corpus takes ~16.5 min per model;
>   only the chosen checkpoint is encoded in full.
> - Full fine-tuning of the shared encoder, as in the paper (a query-side-only model was
>   considered and not chosen).

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

   *Done (with the changed negatives):* `src/train/build_pairs.py`, `scripts/build_train_pairs.py`, `configs/train_pairs_hi.yaml`. One example per (query, relevant passage): 2,467 examples from 1,168 queries in each of N, 50 (584 native / 584 romanized queries) and T, with the same positives and negatives, so the configurations differ only in the query script. One judged negative per example, seeded per query (2,235 distinct). Train shares no query id or text with tune or test. Files in `data/train/` (git-ignored).
2. **Model:** start from `multilingual-e5-base`, the model chosen in Phase 1 (`-small` for quick debugging runs). BGE-M3 does not fit fine-tuning in 8 GB (Phase 1 benchmark).
3. **Training setup for 8 GB VRAM:**
   - sentence-transformers with `CachedMultipleNegativesRankingLoss` (large effective batch with small memory)
   - fp16, max_len ~256 for passages and ~32 for queries
   - fixed seed; config in `configs/train_dense.yaml`

   *Set up and timed (task 2):* `src/train/train_dense.py`, `scripts/train_dense.py`, one config per set-up (`configs/train_dense_50.yaml` first). Scale 100 (τ = 0.01, e5's temperature, Wang et al. 2022 §4.1), learning rate 2e-5 (e5's, Table 11), batch 64 (mini-batches of 16), 3 epochs with a checkpoint per epoch, 10% warm-up, no-duplicates batches, e5 prompts. Needed `datasets` 5.2.0 and `accelerate` 1.15.0 (sentence-transformers' `[train]` extra; added with approval). Timing (10 steps, 50 set-up): e5-small 1.0 s/step, peak 2.64 GB; **e5-base 2.1 s/step, ~4 min for 3 epochs (117 steps), peak 5.95 GB of 8 GB.** Queries are padded per batch, so the 256-token limit only affects passages.

   *Trained (task 3):* e5-base on N, 50 and T (`configs/train_dense_{N,50,T}.yaml`, seed 42), with a checkpoint at the end of each epoch in `models/` (git-ignored). A first run of 3 epochs (3.7–3.8 min each; mean training loss per epoch N 0.560 → 0.304 → 0.169, 50 1.552 → 0.795 → 0.524, T 2.479 → 1.259 → 0.884) put T's best tune checkpoint at the last epoch (`results/tuning/select_checkpoint_hi_3epochs.csv`). *Plan change:* all three were retrained for **6 epochs** (the learning-rate schedule depends on the number of epochs, so all three set-ups were retrained, not only T), 7.6–10.5 min each; T's run was cut off once when the session ended and was redone from the start. N starts with a low loss (native queries are in-domain for e5, which was trained on MIRACL), T with a high one (romanized queries are new to the model). Loss logs in `results/training/`.
4. **Loss (InfoNCE with in-batch + hard negatives):**
   $$\mathcal{L} = -\log \frac{e^{s(q,d^+)/\tau}}{e^{s(q,d^+)/\tau} + \sum_{d^- \in \mathcal{N}(q)} e^{s(q,d^-)/\tau}}$$
   where $s(q,d)$ is the cosine similarity, $\tau$ the temperature and $\mathcal{N}(q)$ the in-batch and hard negatives.
5. **Model selection** on the tune split (nDCG@10 on romanized tune queries), with checkpoints every N steps.

   *Done (task 4):* `scripts/select_checkpoint.py` scores every checkpoint on a fixed candidate pool of 28,740 passages (every passage judged for a tune query plus the top 100 of eight tune runs; pool scores are higher than full-corpus scores and only rank checkpoints). Rule fixed before the run: natural romanized nDCG@10. Chosen (`results/tuning/select_checkpoint_hi.csv`): **N epoch 1** (natural 0.0511, native 0.5655), **50 epoch 5** (0.1933, 0.5778), **T epoch 4** (0.2030, 0.5687); off-the-shelf 0.0310 / 0.5726. All chosen epochs are inside the range. Romanized gains are real but modest (natural ×6–7, still far below native); rule queries stay low (≤ 0.09).
6. *Done (task 5):* the corpus encoded with each chosen checkpoint (`configs/finetuned_dense_{N,50,T}.yaml`, ~11 min each, the baseline's settings) and every query set run on tune and test, raw and with top-1 back-transliteration (36 runs in results.csv). Significance and the table follow in task 6.

   **Evaluate on test** with all query sets, especially:
   - **native** queries → did performance drop? (catastrophic forgetting check)
   - **human-romanized** queries → does it generalize to real spellings, not just the romanizer it was trained on?
7. Small ablation if time allows: romanized-only vs mixed training; with vs without hard negatives.

   *Covered / not done:* romanized-only vs mixed training is the T vs 50 comparison (task 5). The hard-negative ablation was not run.

## Outputs
- `src/train/build_pairs.py`, `src/train/train_dense.py`
- `configs/train_dense.yaml` (*as built:* `configs/train_dense_{N,50,T}.yaml`, one per set-up)
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

**Exit criteria: met.**
- The fine-tuned models beat the off-the-shelf model on romanized test queries (50 and T on natural and rule, T also on the hand-romanized set; significant after Bonferroni over the table's 21 comparisons).
- Native-query performance is reported next to them, with its own significance test.
- Every choice (epochs, checkpoints) was made on tune; test was run once afterwards.

**Test results** (dense e5-base, nDCG@10; `results/tables/dense_ft_hi_test.md` and `dense_ft_native_hi_test.md`; 250 / 250 / 50 / 250 queries)

| Model | natural | rule | human | native |
|---|---|---|---|---|
| off-the-shelf (Phase 4) | 0.0133 | 0.0000 | 0.0106 | **0.6067** |
| fine-tuned N (native queries) | 0.0280 | 0.0025 | 0.0034 | 0.6006 (p = 0.63) |
| fine-tuned 50 (half and half) | 0.1988* | 0.0465* | 0.1122 | 0.5830 (p = 0.060) |
| fine-tuned T (romanized queries) | **0.2259*** | **0.0614*** | **0.1595*** | 0.5468 (p < 0.001*) |
| *Phase 5: off-the-shelf + top-1 back-transliteration* | *0.5608* | *0.4907* | *0.5496* | |
| fine-tuned N + top-1 back-transliteration | **0.5770** (p = 0.21 vs Phase 5) | 0.4904 | **0.5589** | |
| fine-tuned 50 + top-1 back-transliteration | 0.5514 | 0.4886 | 0.5343 | |
| fine-tuned T + top-1 back-transliteration | 0.5221 (p = 0.009 vs Phase 5) | 0.4798 | 0.5304 | |

\* significant against the off-the-shelf model on the same queries after Bonferroni (romanized columns: 21 comparisons; native column: 12).

**Findings**
- **Fine-tuning teaches romanized retrieval, but only part of it.** Natural queries rise from 0.0133 to 0.1988 (50) and 0.2259 (T), i.e. 31–36% of the gap to native closed. Back-transliteration with the unchanged model closes 92% (0.5608). With 1,168 training queries, the model learns only part of the romanized-to-native mapping, while IndicXlit was trained for exactly this mapping.
- **It carries over partly to real typing, hardly to rule spellings.** The model only saw LLM romanizations. On the hand-romanized queries T reaches 0.1595 (p < 0.001; 50: 0.1122, not significant after correction), about 70% of its natural-set score. Uroman spellings (`raassttrapati`) stay near 0.06.
- **Training only on romanized queries makes the model forget.** T loses 0.060 nDCG@10 on native queries (p < 0.001); 50 loses 0.024 (p = 0.060, not significant); N keeps them (p = 0.63).
- **This reproduces the paper's pattern on Hindi** (paper notes, Table 1, BGE-M3 on mMARCO): N does not help romanized queries, T helps them most and costs native performance, and 50 is the compromise that keeps native performance (in the paper it even improves it, here it is level). Our MRR@10 values for comparison: romanized natural 0.0148 → 0.2155 (50) / 0.2394 (T); native 0.6301 → 0.6115 (50) / 0.5617 (T).
- **Fine-tuning does not improve back-transliterated queries.** These are native-script queries, so N fits them best (natural 0.5770 vs 0.5608, p = 0.21, n.s.), and T is significantly worse than the unchanged model (0.5221, p = 0.009). On tune the order of N and Phase 5 is reversed (0.5615 vs 0.5796), which also points to noise.

**Decisions**
- Judged MIRACL negatives instead of BM25 negatives; N / 50 / T with natural romanizations; full fine-tuning of e5-base (scale 100 = e5's τ 0.01, lr 2e-5, batch 64, no-duplicates batches).
- 6 epochs (the 3-epoch run put T's best at the last epoch); checkpoints chosen on a 28,740-passage tune pool by natural nDCG@10: N epoch 1, 50 epoch 5, T epoch 4.
- 64-token queries at search time, as in the Phase 4 baseline.
- Added dependencies `datasets` and `accelerate` (sentence-transformers' `[train]` extra), with approval.

**Surprises**
- Training is cheap (7.6–10.5 min for 6 epochs); encoding the corpus and choosing checkpoints is what costs time.
- Romanized text tokenizes longer than Devanagari: the longest rule query is 56 tokens, so the planned 48 would have cut 4 test queries.
- The session ended twice during long runs; training was redone from the start (checkpoints hold no optimizer state), and encoding and runs were built to resume.

**Limits**
- 1,168 training queries (2,467 examples), far fewer than the paper's mMARCO training set; the gains are bounded by this.
- The hard-negative ablation (judged vs BM25 negatives) was not run.
- Checkpoints were chosen on a candidate pool (all relevant passages included), not on full-corpus tune runs.
- The human set has 50 queries.

**Review gate (approved 2026-11-14)**
- *Off-the-shelf vs fine-tuned on every query set:* fine-tuning helps raw romanized queries (best: T, natural 0.2259), costs native performance with T, and does not beat back-transliteration with the unchanged model on any set. The best dense method stays query-side: back-transliteration, with the off-the-shelf model or the N model (level).
- *Human-romanized: generalization or overfitting to the romanizer?* Partial generalization: T improves real typing significantly (0.0106 → 0.1595), at about 70% of its gain on the LLM romanizations it was trained on, and hardly at all on Uroman spellings.
- *Carried into Phase 8:* the off-the-shelf dense model with top-1 back-transliteration (and the N model as an equal alternative); fine-tuned T on raw romanized queries as a candidate for fusion, since it needs no transliteration.
