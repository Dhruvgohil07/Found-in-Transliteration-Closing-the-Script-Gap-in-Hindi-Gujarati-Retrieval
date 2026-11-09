# Phase 5 — Query-side back-transliteration

**Status:** Done · **Days:** 5–6

## Goal
Close the gap on the **query side**: convert romanized queries back into native script (Latin → Devanagari) before retrieval, and use multiple candidates as query expansion to handle spelling ambiguity.

## Inputs
- Phase 3 romanized query sets
- Phase 4 harness, BM25 index and dense baseline

## Tasks
1. **Install/choose the transliterator (decide on day 1 of this phase):**
   - First choice: **IndicXlit** (AI4Bharat), which supports Hindi and Gujarati and returns top-k candidates.
   - If fairseq won't install on Windows: run it in WSL, or use an HF seq2seq transliteration model.
   _Decided (2026-11-09): **IndicXlit in WSL (Ubuntu 22.04, Python 3.10)**. On Windows, `fairseq` 0.12.2 (needed by `ai4bharat-transliteration`) has no wheel for Python 3.11 and its source build fails (`FileNotFoundError: fairseqersion.txt`). No reliable Hugging Face substitute was found: AI4Bharat's `ai4bharat/IndicXlit` repo holds only the fairseq checkpoints, and the other matches are undocumented community uploads or general LLMs. An LLM (Gemini) was ruled out: SPECS forbids an LLM as the main method, and Gemini produced the natural set. WSL is needed once: the top-k candidates for every romanized word are generated there and frozen to a file; everything else runs on Windows as before._
   _Changed (2026-11-09): **Docker instead of a WSL distro**, since Docker Desktop was already installed (no new system install). The image (`docker/indicxlit/Dockerfile`) pins `python:3.8-slim` by digest, `torch` 1.13.1 CPU, `numpy` 1.23.5, `bitarray` 3.11.0, `fairseq` 0.12.2 and `ai4bharat-transliteration` 1.1.3; Python 3.8 because fairseq 0.12.2 ships ready-made Linux wheels only up to 3.8. `urduhack` (Urdu only, it imports TensorFlow) is replaced by a stub that fails if Urdu is ever requested._
   _Done (2026-11-09): `scripts/backtranslit_candidates.py` collects the 2,441 distinct alphabetic words of the tune and test romanized sets (natural, rule, human) and runs `scripts/indicxlit/generate_candidates.py` in the container: **10 candidates per word (beam 10) with IndicXlit's re-scoring**, $	ext{score} = 0.9\,p_{	ext{model}} + 0.1\,p_{	ext{word}}$ (beam probability and corpus word frequency, each normalised over the candidates; 0 if the candidate is not in the frequency dictionary). IndicXlit's API returns only the ranked strings, so the script records the scores by wrapping its re-scoring step and checks that its ranking matches IndicXlit's own output (0 mismatches). Frozen and versioned: `data/backtranslit/candidates_hi.tsv` (1.9 MB) and `candidates_hi_meta.json`. The model and the Hindi and Gujarati dictionaries (834 MB) are cached in the git-ignored `data/xlit_cache/`; the 850.5 MB dictionary archive was fetched with a resuming `curl` after IndicXlit's own downloader timed out. Examples: `andolan` → आंदोलन, `raajaa` → राजा, `the` → थे; `roy` → रॉय first and राय second, `kahaam` → कहाम first and कहाँ third (where top-k should help). For 67 words no candidate is in the frequency dictionary (all scores 0): their weights must fall back to $p_{	ext{model}}$._
2. **Top-1 back-transliteration:** replace each romanized word with its best native candidate, then run BM25 and dense retrieval.
   _Query building done (2026-11-09): `src/data/backtranslit.py` (`load_candidates`, `candidate_weights`, `top1_query`, `weighted_query`; 9 tests) and `scripts/build_backtranslit_queries.py` with `configs/backtranslit_queries_hi.yaml`, which writes `data/processed/{split}_queries_{set}_bt1.tsv` (git-ignored: rebuilt from the frozen candidates). Top-1 replaces every alphabetic word by its best candidate; numbers, mixed tokens such as `co2` and words without candidates stay as they are; punctuation is dropped. On tune, natural queries come back almost word for word (`punjab ke itihas mein sabse ameer raja kaun tha` → पंजाब के इतिहास में सबसे अमीर राजा कौन था); rule (Uroman) queries mostly too, but its doubled retroflex letters break words (`raassttrapati` → रासस्त्त्रपति instead of राष्ट्रपति)._
   _Weights for top-k: IndicXlit's scores renormalised over the top k, so each word's candidates sum to 1; if all k scores are 0 (no candidate in IndicXlit's frequency dictionary, 67 words, mostly Uroman spellings) the model probabilities are used instead; a 0-score candidate next to scored ones gets weight 0. Low-ranked candidates with a visible word-final virama and an invisible ZWNJ (785 of 24,410) never rank first; Terrier strips the ZWNJ and the remaining form matches nothing, so they are harmless._
   _Runs done on tune (2026-11-09): `configs/backtranslit_top1_bm25.yaml` and `configs/backtranslit_top1_dense.yaml` use exactly the Phase 4 baseline settings, only the query sets change. Test runs wait until step 6, so the choices in steps 4 and 5 are made on tune without seeing test. Tune nDCG@10, with $\text{GapRecovered} = \frac{M_{\text{method}} - M_{\text{roman}}}{M_{\text{native}} - M_{\text{roman}}}$ (new `gap_recovered` in `src/eval/gap.py`):_

   | Method | Set | Native | Romanized | Top-1 | GapRecovered | p (top-1 vs romanized) | p (top-1 vs native) |
   |---|---|---|---|---|---|---|---|
   | BM25 | natural | 0.3954 | 0.0000 | 0.3568 | 90.2% | 7.1e-17 | 0.089 |
   | BM25 | rule | 0.3954 | 0.0000 | 0.3255 | 82.3% | 9.5e-15 | 0.015 |
   | Dense | natural | 0.5724 | 0.0158 | 0.5796 | 101.3% | 2.1e-31 | 0.662 |
   | Dense | rule | 0.5724 | 0.0000 | 0.4988 | 87.1% | 1.1e-23 | 0.010 |

   _On natural queries, top-1 back-transliteration is not significantly different from native for either method (100 tune queries, paired t-test); on rule (Uroman) queries it stays significantly below native._
3. **Top-k expansion (core IR idea):** use the top-k candidates per word as a weighted query, so spelling variants all get a chance to match. Weight each candidate by its transliteration score (normalized per word):
   $$q' = \bigcup_{w \in q} \big\{ (c_{w,j},\; p_{w,j}) \big\}_{j=1}^{k}, \qquad \sum_{j=1}^{k} p_{w,j} = 1$$
   where $c_{w,j}$ is the $j$-th native candidate for word $w$ and $p_{w,j}$ its weight in the BM25 query.
   Tune k ∈ {1, 3, 5} on the **tune split** only.
   _Done on tune (2026-11-09): `retrieve_weighted` in `src/retrieval/bm25.py` (3 tests), `scripts/tune_backtranslit_k.py` with `configs/tune_backtranslit_k_hi.yaml` (table in `results/tuning/tune_backtranslit_k_hi.csv`), runner method `bm25_bt5` with `configs/backtranslit_topk_bm25.yaml`._
   - _**How the weights reach BM25.** Terrier's `#combine` query language cannot parse Devanagari (lexer error), so the query uses the standard `term^weight` syntax; every candidate is first cut by the index tokeniser, so only clean terms reach the parser. Checked on the index: Terrier divides each weight by the largest weight in the query (only ratios matter) and BM25 then scales the term's score by $\frac{(k_3+1)\,x}{k_3+x}$ with $k_3 = 8$, where $x$ is that relative weight (a relative weight of 0.25 became a factor of 0.273, as measured). So the weights are applied almost linearly and their order is always kept._
   - _**Tuning k.** Grid {1, 3, 5} as planned, extended to 7 and 10 when the first best (5) sat at the edge; k = 1 reproduces the top-1 runs exactly (0.3568 / 0.3255). Tune nDCG@10:_

     | Set | k=1 | k=3 | k=5 | k=7 | k=10 |
     |---|---|---|---|---|---|
     | natural | 0.3568 | 0.3475 | 0.3656 | 0.3632 | 0.3712 |
     | rule | 0.3255 | 0.3397 | 0.3475 | 0.3521 | 0.3542 |

   - _Paired tests on tune: no nDCG@10 difference between any two of k = 1, 5, 10 is significant. The significant gain of expansion is **recall**: natural R@1000 +0.073 at k = 5 vs k = 1 (p = 0.003), rule R@100 +0.057 (p = 0.027). k = 10 vs k = 5: no significant difference on any metric (p 0.16-0.69), with k = 5 slightly ahead on R@100 and R@1000._
   - _**Decided: k = 5** for every set: tied with k = 10 on nDCG@10, slightly better recall, half the query terms, inside the planned grid and away from the 10-candidate beam limit (tie-break: the smaller k when the difference is not significant). The human set cannot be tuned (test only) and uses the same k._
4. Handle code-mixed queries: words that are English (e.g. `news`) should not be forced into Devanagari. Try a simple rule (keep words found in an English wordlist) and report whether it helps.
   _Done on tune (2026-11-09): `scripts/backtranslit_english_words.py` + `configs/backtranslit_english_hi.yaml` freeze the English words (`data/backtranslit/english_words_hi.txt`, 122 of 2,441 words); `english_mode` in `src/data/backtranslit.py` (4 tests); `scripts/tune_backtranslit_english.py` + `configs/tune_backtranslit_english_hi.yaml` compare the modes (`results/tuning/tune_backtranslit_english_hi.csv`)._
   - _**Wordlist:** IndicXlit's English word-frequency dictionary (`word_prob_dicts_en.zip`, 1.17 MB, same release). Every common Hinglish word is in it too (`kisan`, `hai`, `mein`), so membership is useless; frequency separates them (Hinglish at most 7.3e-7, loanwords above 1e-5). Rule: at least 4 letters (short words such as `the`, `to`, `is` are also Hinglish words थे, तो, इस) and frequency ≥ 1e-5, chosen from the tune word list. It also catches Indian names common in English text (`delhi`, `bharat`, `punjab`), which Hindi Wikipedia writes in Devanagari; no frequency rule can separate them._
   - _**Why Latin hardly helps here:** Hindi Wikipedia writes loanwords in Devanagari; Latin `cricket` occurs in 20 passages, `train` in 20, `shooting` in 7 (Latin words of 3+ letters appear in 11.2% of the first 50,000 passages, mostly names and references)._
   - _**Result (tune nDCG@10, paired t-test against `none`):**_

     | Method | Set | none | replace | both |
     |---|---|---|---|---|
     | BM25 top-5 | natural | 0.3656 | 0.2455 (−0.120, p < 0.001) | 0.3206 (−0.045, p = 0.003) |
     | BM25 top-5 | rule | 0.3475 | 0.3274 (p = 0.080) | 0.3435 (p = 0.260) |
     | Dense top-1 | natural | 0.5796 | 0.5532 (−0.027, p = 0.017) | 0.5671 (p = 0.166) |
     | Dense top-1 | rule | 0.4988 | 0.5012 (p = 0.517) | 0.4973 (p = 0.320) |

   - _**Decided: `english_mode = none`** (English words are back-transliterated like any word). The plan's rule hurts BM25 badly and dense somewhat; keeping both scripts still hurts BM25, because the Latin word takes query weight away from Devanagari terms that match. Rule queries have almost no English-looking words (Uroman writes `philma` for film)._
5. Evaluate everything on all romanized query sets, next to the Phase 4 baselines.

   _Done (2026-11-09): the three Phase 5 configs were run on test once, with every setting fixed on tune (top-1; k = 5; no English rule). `scripts/method_table.py` with `configs/table_backtranslit_hi.yaml` generates `results/tables/backtranslit_hi_test.md`: each method next to its romanized and native runs, GapRecovered, a paired t-test against the romanized baseline with Bonferroni over the table's 9 comparisons ($p < 0.05/9$), and a paired t-test against native. The script is generic, so later phases add their methods to it. Every method beats its romanized baseline significantly. Top-5 vs top-1 on test (BM25): nDCG@10 not significant (natural +0.023, p = 0.107; rule +0.011, p = 0.337; human −0.019, p = 0.373), R@1000 higher (natural +0.031, p = 0.023; rule +0.038, p = 0.010; human +0.052, p = 0.079), as on tune._

## Outputs
- `src/data/backtranslit.py`
- `data/processed/queries_backtranslit_{natural,rule,human}_top{k}.tsv` (frozen)
- `configs/backtranslit_*.yaml`
- New rows in `results/results.csv`

## Exit criteria
- Top-1 and top-k runs are evaluated on every romanized query set.
- The best k is chosen on the tune split, and test results are reported with significance against BM25-romanized.

## Review gate
- How much of the gap does back-transliteration recover?
- Look at 10 queries where it fails: wrong candidate, English word, or named entity?

## Risks / fallback
- Transliterator setup eats more than half a day → switch to the fallback immediately and note it.

## What we found

**Exit criteria: met.**
- Top-1 and top-k runs are evaluated on every romanized query set (natural, rule, human), with BM25 and, for top-1, the dense model.
- k was chosen on the tune split (k = 5), as was every other setting (top-1, no English rule); test was run once afterwards. Every method is significantly better than its romanized baseline, with Bonferroni over the table's 9 comparisons.

**Test results** (nDCG@10, from `results/tables/backtranslit_hi_test.md`; % = GapRecovered)

$$\text{GapRecovered} = \frac{M_{\text{method}} - M_{\text{roman}}}{M_{\text{native}} - M_{\text{roman}}}$$

| Method | natural (250) | rule (250) | human (50) |
|---|---|---|---|
| BM25 romanized (baseline) | 0.0014 | 0.0017 | 0.0000 |
| BM25 top-1 back-transliteration | 0.3506 (92.0%) | 0.3252 (85.3%) | 0.3479 (88.7%) |
| BM25 top-5 weighted | 0.3737 (98.1%) | 0.3365 (88.3%) | 0.3294 (84.0%) |
| BM25 native | 0.3810 | 0.3810 | 0.3923 |
| Dense romanized (baseline) | 0.0133 | 0.0000 | 0.0106 |
| Dense top-1 back-transliteration | 0.5608 (92.3%) | 0.4907 (80.9%) | 0.5496 (88.2%) |
| Dense native | 0.6067 | 0.6067 | 0.6220 |

- **Back-transliteration closes most of the script gap.** BM25 top-5 on natural queries is statistically indistinguishable from native (p = 0.652); every other run recovers 81-92%.
- **It works on real typing:** on the 50 hand-romanized queries, top-1 recovers 88% for both retrievers.
- **Rule (Uroman) is the hardest set** (81-88%): its doubled retroflex letters (`raassttrapati`) give wrong candidates.
- **Top-k expansion buys recall, not precision:** top-5 vs top-1 changes nDCG@10 insignificantly on tune and test, but raises R@1000 (test natural +0.031, p = 0.023; rule +0.038, p = 0.010).
- **Dense keeps a gap:** natural top-1 is 0.046 below native, significant on test (p < 0.001) though not on the smaller tune split.

**Failure analysis** (`scripts/backtranslit_failures.py`, the 10 tune queries with the largest nDCG@10 drop from native to BM25 top-1, natural set; main cause by our reading):

| Main cause | Queries |
|---|---|
| A valid Devanagari spelling variant | 5 |
| Information lost or ambiguous in the romanization | 3 |
| Named entity | 1 |
| Numeral script | 1 |
| English word | 0 |

- **Most failures are not transliteration errors.** IndicXlit returns a correct spelling, but a different valid one from the native query's: anusvara vs half-nasal consonant (आरंभिक / आरम्भिक, प्रतिबिंब / प्रतिबिम्ब, पंप / पम्प), anusvara vs chandrabindu (आंख / आँख, बांध / बाँध), हुई / हुयी, सम्राट / सम्राट्. Both forms are common in the corpus, and the back-transliterated one is often the more frequent (हुई in 31,232 passages, हुयी in 463; सम्राट 2,360, सम्राट् 359). Even numbers occur in two scripts (२ in 5,770 passages, 2 in 16,868). BM25 matches exact strings, so the passage's spelling decides.
- **The native upper bound has a spelling advantage.** MIRACL's annotators wrote each question after reading its passage, so the native query tends to copy the passage's spelling; a back-transliterated query uses the standard spelling and can miss. Part of the remaining "gap" is therefore spelling variation, not script.
- **Romanization itself loses information:** `pati` could be पति or पटी, `kr` is a shorthand for क्र (कर), `kishmo` dropped the nasal of किस्मों, `bhala` does not mark the long vowel of भाला.
- **This points to Phase 6:** a normalization key that maps spelling variants (anusvara / chandrabindu / half-nasal, numerals) to one form on both the document and the query side addresses the largest failure class directly.

**Decisions**
- Transliterator: IndicXlit v1.0 in a pinned Docker image (Python 3.8), run once; candidates frozen in `data/backtranslit/`.
- Weighted query: IndicXlit's scores renormalised over the top k (model probabilities if all are 0), sent as Terrier `term^weight`; **k = 5**.
- English words: **back-transliterated like any word** (keeping them in Latin hurt on tune).

**Surprises**
- **IndicXlit needed a container:** fairseq does not build on Windows/Python 3.11 and ships Linux wheels only up to Python 3.8; its Urdu dependency imports TensorFlow (replaced by a stub); its downloader cannot resume (the 850.5 MB dictionary was fetched with `curl`).
- **IndicXlit's API hides its scores;** the container script records them and checks that they reproduce IndicXlit's ranking (0 mismatches).
- **Terrier's `#combine` cannot parse Devanagari,** and `term^weight` weights are relative to the largest weight and saturate slightly (factor $\frac{9x}{8+x}$).
- **Hindi Wikipedia writes English loanwords in Devanagari** (Latin `cricket` in 20 passages), so keeping English words in Latin lowered BM25 nDCG@10 by 0.120 on tune.

**Limits**
- The failure labels are our reading of 10 queries, not an annotated sample.
- Top-k expansion is BM25 only; the dense model uses top-1 queries.
- The human set has 50 queries, so its differences between methods are rarely significant.

**Review gate (approved 2026-11-09)**
- Back-transliteration recovers most of the gap on test: BM25 top-5 98.1% (natural, tied with native), 88.3% (rule), 84.0% (human); dense top-1 92.3%, 80.9%, 88.2%; all significant against the romanized baselines.
- The failures show that the main remaining cause is Devanagari spelling variation (anusvara / chandrabindu / half-nasal, numeral script), not wrong transliteration; this becomes an input to Phase 6 (normalization on both sides).
- Methods carried forward as baselines for later phases: BM25 top-5 weighted (`configs/backtranslit_topk_bm25.yaml`) and dense top-1 (`configs/backtranslit_top1_dense.yaml`).
