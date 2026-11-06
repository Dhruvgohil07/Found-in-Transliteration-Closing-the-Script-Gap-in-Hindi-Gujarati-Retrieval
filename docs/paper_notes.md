# Reference Paper Notes

**Paper:** Andreas Chari, Iadh Ounis, Sean MacAvaney. *Lost in Transliteration: Bridging the
Script Gap in Neural IR.* SIGIR 2025 (short paper, 6 pages).
DOI: [10.1145/3726302.3730226](https://doi.org/10.1145/3726302.3730226) ·
Code: <https://github.com/andreaschari/transliterations> (footnote 1) ·
License: CC BY 4.0.

Every fact below is taken from the paper, with its section, table or footnote. Anything that is
our own reading or calculation is marked **(ours)**.

---

## 1. Research questions (§1)

- **RQ1:** How robust are existing neural IR methods to transliterated queries?
- **RQ2:** Can the script gap be bridged with "transliterate-train"?
- **RQ3:** Does transliterate-train generalise to out-of-domain collections?

The paper calls the difference in effectiveness between the native-script and transliterated
versions of the same queries the **"script gap"** (§1).

## 2. Datasets and languages (§1, §4)

| Role | Collection | Languages | Query sets used |
|---|---|---|---|
| In-domain train + eval | mMARCO/v2 (machine-translated MS MARCO passages) | Chinese, Russian | `train` (fine-tuning), `dev/small` (evaluation) |
| Out-of-domain eval | neuCLIR/1 (Common Crawl news) | Chinese, Russian | `trec-2022` + `trec-2023`, combined |

- Data was accessed through `ir_datasets` (§4).
- Why these two languages (§4): Chinese script has **no overlap** with Latin, while Russian Cyrillic
  **overlaps a lot** with Latin. They are two extremes.
- Documents always stay in the native script. Only queries are transliterated (§3, §4).

## 3. How the romanized queries were produced (§1, §4)

- **Tool:** Uroman, a rule-based "universal romanizer" (ref. [9], Hermjakob et al., ACL 2018 demo).
- Applied to the `dev/small` and `train` queries of both mMARCO languages, and to the neuCLIR
  `trec-2022` and `trec-2023` queries (§4).
- The romanizations are **machine-generated**, not typed by real users. The paper describes this
  as queries "machine-translated into a Romanized script" (§2).
- **Chinese tokenisation:** pre-tokenising Chinese queries before transliteration (via
  PyTerrier's Anserini interface, footnote 3) gave **inconsistent results** between BGE-M3 and mT5.
  The main results therefore use **non-tokenised** queries, and tokenisation is left for future
  work (§4.1, §4.2).

## 4. Models, training and baselines (§3, §4)

**Pipeline**
- **Retriever:** BGE-M3, using **only its single-vector (dense) component** (§4).
- **Reranker:** mT5, written as `BGE-M3 >> mT5`, i.e. mT5 reranks BGE-M3's results (Table 1).
- **Tools:** PyTerrier for retrieval, `ir-measures` for evaluation (§4).
- **No lexical baseline:** there is no BM25 or other lexical baseline. The baselines are the
  off-the-shelf BGE-M3 and mT5 (§4).

**Transliterate-train (§3).** This is a variant of *translate-train*, where a large judged
dataset is machine-translated into another language (Nair et al., ref. [20]). Here the translated
mMARCO training queries are additionally **romanized**, and the models are fine-tuned on
(query, native document) pairs in which the query is native or romanized.

**Three fine-tuning configurations (§4).** In all three the documents stay native.

| Name | Training queries |
|---|---|
| **N** | native-script only |
| **50** | even split of native and transliterated |
| **T** | transliterated only |

Both BGE-M3 and mT5 are fine-tuned in each configuration.

## 5. Metrics and statistics (§4)

| Collection | Metrics |
|---|---|
| mMARCO/v2 `dev/small` | **MRR@10**, **R@1000** |
| neuCLIR (trec-2022 + 2023) | **nDCG@20**, **R@1000** |

- **Significance tests:** each fine-tuned configuration is compared with its own baseline
  (BGE-M3 or mT5) on the same query set, with **Bonferroni correction**. The symbol † marks a
  significant result. The paper does **not** name the test or the significance level (§4).

## 6. Headline results

Notation: (N) = native queries, (T) = Uroman-transliterated queries.

### Table 1: mMARCO/v2, in-domain

| Model | zh (N) MRR@10 | zh (N) R@1000 | zh (T) MRR@10 | zh (T) R@1000 | ru (N) MRR@10 | ru (N) R@1000 | ru (T) MRR@10 | ru (T) R@1000 |
|---|---|---|---|---|---|---|---|---|
| BGE-M3 | 0.2342 | 0.8940 | 0.0078 | 0.0550 | 0.2444 | 0.8998 | 0.1244 | 0.6589 |
| BGE-M3 (N) | 0.2732† | 0.9125† | 0.0071 | 0.0524 | 0.2838† | 0.9137† | 0.1238 | 0.5925† |
| BGE-M3 (50) | 0.2642† | 0.8991 | 0.1382† | 0.6807† | 0.2770† | 0.9117† | 0.2633† | 0.8895† |
| BGE-M3 (T) | 0.2112† | 0.7967† | 0.1608† | 0.7255† | 0.2785† | 0.9105† | 0.2679† | 0.8963† |
| BGE-M3 >> mT5 | 0.2441 | 0.8940 | 0.0081 | 0.0550 | 0.2552 | 0.8998 | 0.1948 | 0.6589 |
| BGE-M3 (N) >> mT5 (N) | 0.2503 | 0.9125† | 0.0070 | 0.0524 | 0.2522 | 0.9137† | 0.1192† | 0.5925† |
| BGE-M3 (50) >> mT5 (50) | 0.2520† | 0.8991 | 0.1105† | 0.6807† | 0.2533 | 0.9117† | 0.2339† | 0.8895† |
| BGE-M3 (T) >> mT5 (T) | 0.2070† | 0.7967† | 0.1145† | 0.7255† | 0.2662† | 0.9105† | 0.2522† | 0.8963† |

### Table 2: neuCLIR, out-of-domain (trec-2022 + trec-2023 combined)

Only the mixed (50) configuration was evaluated here (§4.3).

| Model | zh (N) nDCG@20 | zh (N) R@1000 | zh (T) nDCG@20 | zh (T) R@1000 | ru (N) nDCG@20 | ru (N) R@1000 | ru (T) nDCG@20 | ru (T) R@1000 |
|---|---|---|---|---|---|---|---|---|
| BGE-M3 | 0.4279 | 0.7500 | 0.0028 | 0.0251 | 0.4373 | 0.7460 | 0.2031 | 0.5018 |
| BGE-M3 (50) | 0.3886† | 0.6466† | 0.0698† | 0.2488† | 0.3924† | 0.6403† | 0.3293† | 0.6070† |
| BGE-M3 >> mT5 | 0.4237 | 0.7500 | 0.0028 | 0.0251 | 0.4418 | 0.7460 | 0.2970 | 0.5018 |
| BGE-M3 (50) >> mT5 (50) | 0.3380† | 0.6466† | 0.0605† | 0.2488† | 0.3541† | 0.6503† | 0.2851 | 0.6070† |

### Findings in the paper's own words (summarised)

- **RQ1 (§4.1):** with transliterated queries, off-the-shelf BGE-M3's MRR@10 drops by **97%**
  (Chinese) and **49%** (Russian). Even Russian, whose script overlaps with Latin, loses about half
  its effectiveness. The mT5 reranker shows the same pattern.
- **RQ2 (§4.2):**
  - **N** improves native queries but does nothing for transliterated ones.
  - **50** significantly improves transliterated queries in both languages, and native
    performance is kept. For Russian, native and transliterated scores become almost the same;
    for Chinese a smaller gap remains.
  - **T** gives BGE-M3's best transliterated scores, but Chinese native performance drops.
  - The paper concludes that **50 is the only configuration that keeps native performance while
    making both models robust to transliteration**.
- **RQ3 (§4.3):**
  - Out of domain, **50** gives significant transliterated gains for Russian, but native
    performance drops considerably.
  - For Chinese, native performance drops and the transliterated gains are only modest.
  - A qualitative check of 10 `trec-2022` queries found that transliteration loses query
    **nuance**, e.g. qid 32, "treatment of peanut allergy", returns generic medical documents.
  - Most of the 10 sampled queries share **three or fewer** of their top-10 documents across the
    two scripts.

### Our observations on the tables (ours)

- In both tables, the `>> mT5` rows have the **same R@1000** as the retriever they rerank. That
  fits mT5 reordering BGE-M3's candidate list rather than adding new documents. It also shows that
  recall at a deep cutoff measures the first-stage retriever, not the reranker.
- **One exception:** in Table 2, Russian (N) R@1000 is **0.6403** for BGE-M3 (50) but **0.6503**
  for BGE-M3 (50) >> mT5 (50). This breaks the pattern above, so it may be a typo in the paper.
  We copy the numbers as printed.
- The reranker does **not** always help. In Table 2, `BGE-M3 (50) >> mT5 (50)` scores below
  `BGE-M3 (50)` on every nDCG@20 column.

## 7. What the paper does not report

These details would be needed for an exact reproduction but are not in the paper:

- Fine-tuning hyperparameters: learning rate, batch size, epochs, loss, negatives, temperature.
- How many training queries were used, and how the 50/50 split was sampled.
- The mT5 checkpoint and the reranking depth (how many BGE-M3 results mT5 rescores).
- The Uroman version and options.
- The significance test used, and its α.

The code repository (footnote 1) may contain some of these. We have not checked it, so we assume
none of them.

## 8. How this maps to our project (ours, for the review gate)

| Aspect | Paper | Our plan (SPECS) | Note |
|---|---|---|---|
| Languages | Chinese, Russian | Hindi (core), Gujarati (Phase 10) | Hindi in Devanagari is closer to the Chinese case: no shared letters with Latin |
| Collection | mMARCO/v2 + neuCLIR | MIRACL-hi (candidate) | Open question: does mMARCO/v2 have Hindi? (Phase 2) |
| Romanized queries | Uroman, rule-based, machine-generated | natural "Hinglish" (main) + rule-based (contrast) + small hand-made set | Uroman is a candidate for our rule-based set (Phase 3) |
| Lexical baseline | none | BM25 native + romanized (Phase 4) | We add what the paper leaves out |
| Dense model | BGE-M3 (single-vector) | multilingual-e5-base (main), -small for debugging | **Decided in Phase 1:** BGE-M3 needs more than 8 GB to fine-tune, so we use it off-the-shelf only, as a reference point |
| Reranker | mT5 | none planned | Out of scope unless time allows |
| Fine-tuning | N / 50 / T configurations | romanized query → native passage with BM25 hard negatives (Phase 7) | Mirroring N / 50 / T would make our Phase 7 directly comparable |
| Gap-closing beyond training | none | back-transliteration, document-side normalization, hybrid fusion | Our main addition |
| Metrics | MRR@10 + R@1000; nDCG@20 + R@1000 | nDCG@10, MRR@10, R@100, plus R@1000 | **Decided in Phase 1:** R@1000 added as a secondary metric so recall is comparable with the paper |
| Significance | Bonferroni-corrected, test not named | paired t-test + multiple-comparison correction | Compatible |
