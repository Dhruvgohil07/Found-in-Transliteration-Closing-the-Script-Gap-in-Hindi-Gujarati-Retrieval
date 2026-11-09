# Phase 4 — Baselines & evaluation harness ★ milestone

**Status:** Done · **Days:** 3–4

## Goal
Build the evaluation harness that every later phase uses, and **reproduce the script gap**: show how much retrieval quality drops when native queries are replaced by romanized ones.

## Inputs
- Phase 2 corpus, splits and qrels
- Phase 3 query sets (native, natural, rule, human)

## Tasks
1. **Evaluation harness** (`src/eval/`), using `ir_measures`:
   - nDCG@10, MRR@10 and Recall@100 (plus Recall@1000, secondary) for a run file + qrels
   - per-query scores saved alongside the averages
   - paired significance test between two runs (paired t-test, with a Bonferroni correction when comparing many runs; SPECS §7.2)
   - append one row per run to `results/results.csv`: method, query set, split, config path, seed, metrics
   _Done (2026-11-08): `src/eval/evaluate.py` and `src/eval/significance.py`, 13 tests with hand-computed values. Two additions: **Judged@10** is logged as a diagnostic, since MIRACL judged only about 10 passages per query (Phase 2 review gate); and `n_queries` is logged, because every query in the qrels counts and a query without results scores 0._
2. **BM25 index** of the native corpus with PyTerrier (record the tokenizer/stemmer settings in the config).
   _Done (2026-11-09): `src/retrieval/bm25.py`, `scripts/build_bm25_index.py`, `configs/index_bm25_hi.yaml`; 4 tests on a toy index. Each passage is indexed as title + text with the `utf` tokeniser and **no stemming and no stopword removal**: Terrier's defaults are English (Porter turns `roy` into `roi`; the stopword list drops `the`, which is a Hindi verb in romanized text) and do nothing for Devanagari. The index (`data/indexes/hi_native/`, 213 MB, git-ignored) holds all 506,264 passages, 758,041 distinct terms and 35,216,984 tokens (average 69.6 per passage), and builds in about 4 minutes. A first build was interrupted from outside and left an incomplete index; it was deleted and rebuilt._
3. **Baseline runs** on the test split:
   _BM25 half done (2026-11-09): `scripts/run_baseline.py` with `configs/baseline_bm25.yaml`; runs on tune (for later tuning) and test. Three findings changed the setup:_
   - _**Queries are tokenised before searching.** Terrier's query parser read `co2 (carbon) : test` as an empty query and returned nothing, without an error. Each query now goes through the same `utf` tokeniser as the passages (`pt.rewrite.tokenise`)._
   - _**IDF floored at 0.** Terrier's classic IDF $\lograc{N - df + 0.5}{df + 0.5}$ is negative for terms in more than half of all passages: के (75.7%), में, है, की, और. Such terms are dropped from the query. On tune, native nDCG@10 went from 0.3290 to 0.3508 (paired t-test p = 0.027; better on 19 queries, worse on 9); romanized runs are unchanged. Comparison runs: `configs/baseline_bm25_classic.yaml` and `configs/baseline_bm25_idf0.yaml` (tune only)._
   - _**k1 and b tuned on tune.** A grid over k1 ∈ {0.6, 0.9, 1.2, 1.5, 2.0} and b ∈ {0.1, …, 0.9} (`configs/tune_bm25_hi.yaml`, table in `results/tuning/tune_bm25_hi.csv`) picked **k1 = 1.2, b = 0.3** by nDCG@10 (tune 0.3954, against 0.3508 at Terrier's default b = 0.75). b = 0.1 and 0.2 were added after the first best (0.3) sat at the grid edge. Before tuning, one test run at the default parameters had been made as a first sanity check; the parameters were chosen on tune only._
   - _**Sanity check against published numbers.** MIRACL (arXiv:2210.09984, Table 2, dev set, Anserini/Lucene with the Hindi analyzer) reports Hindi BM25 nDCG@10 0.458 and Recall@100 0.868. Ours on the 250 test queries (a subset of dev) is 0.381 and 0.752: the same range, but lower. Likely reasons: no Hindi normalization or stemming (Lucene's analyzer has both), different BM25 implementations and parameters, and MIRACL's judgments were pooled from a Lucene BM25 among other systems. Accepted for now; Hindi normalization is a possible later improvement._
   - _**Run files** (`results/runs/*.trec`, 1-17 MB each) are git-ignored and rebuilt from their configs; `results/results.csv` and `results/per_query/*.csv` are versioned._
   _Dense half done (2026-11-09): `src/retrieval/dense.py`, `scripts/encode_corpus.py`, `configs/baseline_dense.yaml` (multilingual-e5-base, fp16, `query: ` / `passage: ` prefixes, passages cut at 256 tokens, queries at 64, queries **as typed**, no lower-casing). The corpus was encoded once in 16.5 min on the RTX 4060 and cached in 11 resumable shards (`data/embeddings/hi_e5base/`, 778 MB float16, git-ignored); search is exact (dot product in float32 on the GPU), and both splits run in under 30 s._
   - _**The model's default prompts are empty**, so the e5 prefixes are passed explicitly; a test checks they are applied exactly once._
   - _**Sanity check against published numbers.** The mE5 technical report (arXiv:2402.05672, Table 6) gives mE5-base on MIRACL-hi dev nDCG@10 58.4 and Recall@100 92.6; ours on the 250 test queries (a subset of dev) is 0.607 and 0.941, so the pipeline matches._
   - _**mE5 was fine-tuned on MIRACL** (same report, Table 2: MIRACL 40k and Mr. TyDi 50k in the supervised mixture). The native dense baseline is therefore in-domain, which partly explains why it beats BM25 by so much (0.607 vs 0.381). It also matters for Phase 7: the native MIRACL training queries are not new to this model, only their romanized versions are._
   - _**Exact reproducibility needs a fixed search chunk size**: different matrix shapes sum in a different order and change scores in the last float32 bit (~1e-7), not the ranking. The chunk size is fixed (64)._
   | Run | Query set | Role |
   |---|---|---|
   | BM25 | native | upper bound |
   | BM25 | natural / rule / human | shows the gap |
   | Dense (off-the-shelf, e.g. `multilingual-e5-base`) | native | dense upper bound |
   | Dense (off-the-shelf) | natural / rule / human | dense gap |
4. **Gap table:** relative drop per metric, e.g. $\Delta = \frac{M_{\text{native}} - M_{\text{roman}}}{M_{\text{native}}}$.
   _Done (2026-11-09): `src/eval/gap.py` (4 tests), `scripts/gap_table.py`, `configs/gap_table_hi.yaml`; the generated table is `results/tables/gap_hi_test.md`. Each romanized set is compared with native on the same queries (the human set's native column uses those 50 queries), with a paired t-test and Bonferroni over the 3 romanized sets ($p < 0.05/3$). Every drop is significant: 94.6-100% for BM25 and 85.9-100% for dense, across all four metrics._
5. **Reproducibility check:** re-run one baseline from scratch with the same config and seed and confirm identical metrics.
   _Done (2026-11-09) for both baselines with `scripts/check_reproducibility.py`, which redoes everything in a temporary folder (`results/` and `results.csv` untouched, checked by file hash) and compares every run file byte for byte and every per-query score. **BM25:** index rebuilt from the corpus, then all 7 runs: all byte-identical. **Dense:** the first corpus shard re-encoded on the GPU (50,000 of 50,000 vectors bit-identical to the cache), then all 7 runs: all byte-identical. The runner's loop became `run_all()` in `scripts/run_baseline.py` so the check runs exactly the same code._

### Metric definitions (for the report)
$$\text{nDCG@}k = \frac{1}{\text{IDCG@}k}\sum_{i=1}^{k} \frac{2^{rel_i}-1}{\log_2(i+1)}$$
$$\text{MRR@}k = \frac{1}{|Q|}\sum_{q \in Q} \frac{1}{\text{rank}_q}\quad(\text{0 if no relevant doc in top } k)$$
$$\text{Recall@}k = \frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|}$$

## Outputs
- `src/eval/evaluate.py`, `src/eval/significance.py`
- `src/retrieval/bm25.py`, `src/retrieval/dense.py`
- `configs/baseline_*.yaml`
- `results/runs/*.trec`, `results/per_query/*.csv`, `results/results.csv`
- Gap table in "What we found"

## Exit criteria
- All baseline runs are evaluated and logged.
- The reproducibility check gives identical numbers.
- BM25 native→native is in a plausible range for this collection (compared with **verified** published numbers).

## Review gate
- Is the gap clearly visible and does its trend match the paper?
- Are the numbers sane? If BM25-native looks off, fix it here before building anything on top.

## Risks / fallback
- Dense encoding of the full corpus is slow → encode once, cache embeddings to disk (git-ignored), and reuse them in later phases.

## What we found

**Exit criteria: met.**
- All baseline runs are evaluated and logged: 20 rows in `results/results.csv` (BM25 and dense on tune and test, plus the two BM25 comparison variants on tune), each with per-query scores in `results/per_query/`.
- The reproducibility check gives identical results, stronger than identical metrics: every run file is byte-identical after rebuilding the BM25 index and re-encoding a dense shard from scratch (task 5).
- BM25 native is in a plausible range: test nDCG@10 0.381 and Recall@100 0.752, against MIRACL's published Lucene BM25 of 0.458 and 0.868 on all of dev (task 3 note: lower, with known reasons). The dense baseline matches its published numbers closely (0.607 / 0.941 against the mE5 report's 58.4 / 92.6).

**The script gap on the test split** (from `results/tables/gap_hi_test.md`, generated by `scripts/gap_table.py`)

$$\Delta = \frac{M_{\text{native}} - M_{\text{roman}}}{M_{\text{native}}}$$

| Method | Query set | Queries | nDCG@10 | MRR@10 | R@100 | R@1000 | Δ nDCG@10 |
|---|---|---|---|---|---|---|---|
| BM25 | native | 250 | 0.3810 | 0.4016 | 0.7524 | 0.8888 | |
| BM25 | roman_natural | 250 | 0.0014 | 0.0007 | 0.0233 | 0.0483 | 99.6% |
| BM25 | roman_rule | 250 | 0.0017 | 0.0010 | 0.0100 | 0.0170 | 99.5% |
| BM25 | roman_human | 50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0% |
| Dense e5-base | native | 250 | 0.6067 | 0.6301 | 0.9412 | 0.9973 | |
| Dense e5-base | roman_natural | 250 | 0.0133 | 0.0148 | 0.0551 | 0.1405 | 97.8% |
| Dense e5-base | roman_rule | 250 | 0.0000 | 0.0000 | 0.0000 | 0.0130 | 100.0% |
| Dense e5-base | roman_human | 50 | 0.0106 | 0.0133 | 0.0460 | 0.1067 | 98.3% |

The human set's Δ compares with the native scores of the same 50 queries (BM25 0.3923, dense 0.6220 nDCG@10). Every drop, on every metric, is significant (paired t-test, Bonferroni over the 3 romanized sets, $p < 0.05/3$).

**Does the trend match the paper?** Yes. The paper's off-the-shelf BGE-M3 loses 97% of its MRR@10 on Uroman-transliterated Chinese queries (0.2342 → 0.0078, `docs/paper_notes.md` §6, Table 1) and 49% on Russian, whose script is closer to Latin. Hindi in Devanagari, like Chinese, shares no letters with Latin script, and our dense MRR@10 drops by 97.7% (natural) and 100% (rule, the paper's tool). Language, collection and model all differ from the paper's, so the pattern is what we reproduce, not the numbers.

**What the gap looks like in detail**
- **BM25 has nothing to match.** A Latin-script query shares no terms with Devanagari passages; the few non-zero scores come from Latin words that do appear in Hindi Wikipedia (`co2`, English names).
- **Dense keeps a little recall.** R@1000 stays at 0.14 (natural) and 0.11 (human), against 0.05 at most for BM25: the model places some relevant passages in its top 1000 but almost never near the top. Later rank-based methods (fusion, Phase 8) have a little to work with here; BM25 has almost nothing.
- **Rule (Uroman) is the hardest set for dense** (nDCG@10 0.000): its spelling (`raajaa ... kahaam`) is further from text the model has seen than casual Hinglish is.
- **The human set behaves like the natural set** for dense (Δ 98.3% vs 97.8%), so the natural set is a fair stand-in for real typing as far as the off-the-shelf model is concerned.

**Setup decisions made in this phase** (details in the task notes)
- BM25 indexes title + text with the `utf` tokeniser and **no stemming or stopwords**; queries pass through the same tokeniser before searching.
- BM25 **floors IDF at 0** (drops terms in more than half of all passages) and uses **k1 = 1.2, b = 0.3**, tuned on tune.
- The dense model gets queries **as typed**, with the e5 `query: ` / `passage: ` prefixes passed explicitly.
- Run files are git-ignored; results, per-query scores and generated tables are versioned.

**Surprises**
- **Terrier failed silently twice:** queries with operator characters were parsed as empty, and the classic IDF gave common Hindi words negative weights (के in 75.7% of passages).
- **multilingual-e5 was fine-tuned on MIRACL** (mE5 report, Table 2), so the native dense baseline is in-domain. This partly explains its large lead over BM25 (0.607 vs 0.381), and for Phase 7 it means only the romanized training queries are new to the model.
- **The judgments are shallow:** Judged@10 is 0.29 (BM25) and 0.38 (dense) for native queries, so most of each top 10 has no judgment and counts as non-relevant. Absolute scores are therefore underestimates, the same for every method. For romanized runs Judged@10 is close to 0: what they retrieve was never judged, and is very unlikely to be relevant.
- **GPU encoding was bit-for-bit deterministic** (50,000 of 50,000 re-encoded vectors identical); only the search chunk size had to be fixed to keep scores identical in the last float32 bit.

**Limits**
- Our BM25 is below MIRACL's Lucene BM25 (0.381 vs 0.458 nDCG@10); Hindi normalization and stemming are an open, possible improvement.
- Only one off-the-shelf dense model (e5-base); BGE-M3, the paper's model, was left out to save time (about an hour of encoding).
- The 50-query human set gives a check, not the main evidence.

**Review gate (approved 2026-11-09)**
- The gap is clearly visible (94.6-100% for BM25, 85.9-100% for dense, every drop significant) and its trend matches the paper (dense MRR@10 −97.7% on natural, −100% on rule, against BGE-M3's −97% on Chinese).
- The numbers are sane: dense matches the mE5 report; BM25 is below MIRACL's Lucene BM25 for documented reasons, and is accepted as the lexical baseline. Hindi normalization stays an open, possible improvement.
- The baselines for every later phase: `configs/baseline_bm25.yaml` (native test nDCG@10 0.381) and `configs/baseline_dense.yaml` (0.607); every method is reported next to them.
