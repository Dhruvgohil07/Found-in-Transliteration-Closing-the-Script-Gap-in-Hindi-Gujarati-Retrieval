# Phase 1 — Setup & groundwork

**Status:** Not started · **Days:** 1

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
4. **Write a `.gitignore`** covering data, indexes, checkpoints (`*.pt`, `*.bin`, `*.safetensors`, `*.faiss`), `.env`, and caches.
5. **Write shared utilities** in `src/utils/`:
   - `set_seed(seed)` for Python, NumPy and torch
   - YAML config loading
   - read/write for TREC run files (`qid Q0 docid rank score tag`)
6. **Write a smoke-test script** that checks imports, `torch.cuda.is_available()`, and PyTerrier/Java init.
7. **Pick the starting dense model** (candidates: `intfloat/multilingual-e5-small` / `-base`), taking the paper's choice and the 8 GB VRAM limit into account.

## Outputs
- `requirements.txt`, `.gitignore`, repo skeleton
- `src/utils/` (seed, config, run-file I/O)
- `scripts/check_env.py` (smoke test)
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
_(fill in at the end of the phase)_
