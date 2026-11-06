# Phase 1 — Setup & groundwork

**Status:** Done (review gate approved 2026-11-06) · **Days:** 1

## Goal
Get a working, reproducible environment and repo skeleton, and pin down exactly what the reference paper did, so later phases don't rest on assumptions.

## Inputs
- `CLAUDE.md` (working rules, commit rules)
- `docs/SPECS.md` (tech stack, repo structure, conventions)
- Reference paper: "Lost in Transliteration", Chari, Ounis & MacAvaney, SIGIR 2025

## Tasks
1. **Read the paper and write down its exact setup** in `docs/paper_notes.md`:
   - datasets and languages
   - how romanized queries were produced
   - retrieval models and baselines
   - metrics and cutoffs
   - headline results (copied from the paper, with table/figure number)
2. **Install packages into the conda env `script-gap`** (Python 3.11, already created) from a pinned `requirements.txt`:
   the "decided" libraries from `docs/SPECS.md` §3 (python-terrier, ir_datasets, ir_measures, ranx, sentence-transformers, torch CUDA build, indic_transliteration, pandas, pyyaml, matplotlib, jupyter, pytest, ruff).
3. **Create the repo skeleton** from `docs/SPECS.md` §4: `src/{data,retrieval,train,eval,utils}`, `scripts/`, `configs/`, `results/`, `models/` and `data/` (both git-ignored), `app/`, `notebooks/`; `docs/` already exists.
   _Change (2026-11-06): added `tests/` for the pytest unit tests, which SPECS §3 lists but §4 had no folder for; SPECS §4 now also records the import convention (`src` as a package, run from the repo root)._
4. **Write a `.gitignore`** covering data, indexes, checkpoints (`*.pt`, `*.bin`, `*.safetensors`, `*.faiss`), `.env`, and caches.
5. **Write shared utilities** in `src/utils/`:
   - `set_seed(seed)` for Python, NumPy and torch
   - YAML config loading
   - read/write for TREC run files (`qid Q0 docid rank score tag`)
6. **Write a smoke-test script** that checks imports, `torch.cuda.is_available()`, and PyTerrier/Java init.
   _Change (2026-11-06): also checks installed versions against requirements.txt, runs an fp16 matmul on the GPU, and builds a tiny Hindi BM25 index, because Terrier's default tokeniser was found to drop all Devanagari (see progress.md, Known issues)._
7. **Pick the starting dense model** (candidates: `intfloat/multilingual-e5-small` / `-base`), taking the paper's choice and the 8 GB VRAM limit into account.
   _Change (2026-11-06): measured instead of assumed, with `scripts/bench_dense_models.py` (placeholder Hindi text, 256-token passages, 32-token queries; hardware fit only, not retrieval quality). BGE-M3, the paper's model, was added as a third candidate. Measured on the RTX 4060 (8 GB):_

   | Model | Params | Dim | Encoding (fp16, batch 64) | One fine-tuning step, peak memory (batch 8 / 16 / 32) |
   |---|---|---|---|---|
   | multilingual-e5-small | 118M | 384 | 874 passages/s, 0.37 GB | 2.21 / 2.23 / 2.24 GB |
   | multilingual-e5-base | 278M | 768 | 415 passages/s, 0.82 GB | 5.22 / 5.24 / 5.26 GB |
   | BGE-M3 | 568M | 1024 | 143 passages/s, 1.45 GB | 10.58 / 10.59 / 11.76 GB (more than the card has) |

   _Decided: e5-base is the main model, e5-small is for quick debugging runs, and BGE-M3 is used off-the-shelf only, as a reference point to the paper. Metric cutoffs were also decided here: nDCG@10, MRR@10 and R@100, plus R@1000 as a secondary metric for comparison with the paper, so runs are now 1000 deep (SPECS §5.4, §7.1)._

## Outputs
- `requirements.txt`, `.gitignore`, repo skeleton
- `src/utils/` (seed, config, run-file I/O)
- `scripts/check_env.py` (smoke test)
- `scripts/bench_dense_models.py` (GPU fit of the candidate dense models)
- `docs/paper_notes.md` (paper setup and headline results)

## Exit criteria
- Smoke test passes: imports OK, GPU visible to torch, PyTerrier starts on Java 17.
- Paper notes are complete, with every number traceable to a table/section in the paper.

## Review gate
- Go through the paper notes: do we agree on the evaluation setup we're reproducing?
- Confirm the starting dense model.

## Risks / fallback
- A library fails on Windows/Python 3.11 → try the conda-forge package, or WSL for that piece only; record what was done.

## What we found

**Exit criteria: met.** `python -m scripts.check_env` passes 6/6 (Python 3.11, pinned versions,
imports, fp16 on the RTX 4060, Java 17 + Terrier 5.11, Hindi BM25 end to end), and every number
in `docs/paper_notes.md` is traced to a table or section of the paper (Tables 1–2 checked number
by number).

**Surprises**
- **Terrier silently drops Devanagari.** Its default English tokeniser, and `tokeniser="UTF"`
  (wrong case), both build a Hindi index with 0 terms and only a warning. Every Hindi/Gujarati
  index must use `tokeniser="utf"`. Without the end-to-end check, every Hindi BM25 number from
  Phase 4 on could have been quietly wrong.
- **GPU memory overflow on Windows does not always raise OOM.** The driver spilled BGE-M3's
  fine-tuning step into system RAM, so it kept running at 0% GPU use instead of failing. A peak
  above 8 GB means "too slow to use", not "fits".
- **Fine-tuning memory barely depends on batch size** for these models (e.g. e5-base: 5.22 GB at
  batch 8, 5.26 GB at batch 32): the weights, gradients and AdamW state dominate, not the batch.
- **The paper has no lexical baseline** and leaves out its training hyperparameters, Uroman
  options and significance test (`paper_notes.md` §7). Our BM25 baselines fill the first gap.

**Decisions**
- Dense model: `multilingual-e5-base` (main), `-small` for debugging, BGE-M3 off-the-shelf only
  (task 7 table above).
- Metrics: nDCG@10, MRR@10, R@100, plus R@1000 as secondary; runs are 1000 deep.
- Reproduction scope: the paper's setup on Hindi (native vs romanized versions of the same
  queries against native documents), compared by pattern, not by number
  (`paper_notes.md` §8).
- Significance: paired t-test with Bonferroni correction, as in the paper (SPECS §7.2).
- Missing paper details are looked up in its code repository at the start of the phase that
  needs them (Phase 3, Phase 7).

**Time.** Planned as 1 working day; the recorded dates run from 2026-10-27 (task 1) to
2026-11-06 (tasks 2–7 and the review gate).
