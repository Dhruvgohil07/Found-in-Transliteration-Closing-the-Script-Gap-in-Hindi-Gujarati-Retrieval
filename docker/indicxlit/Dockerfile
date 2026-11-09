# IndicXlit (AI4Bharat) for Latin -> Devanagari back-transliteration (Phase 5).
#
# fairseq 0.12.2, which IndicXlit needs, does not build on Windows with Python
# 3.11 and ships ready-made Linux wheels only for Python 3.6-3.8, so it runs here
# in a throwaway Linux container. It is used once: the candidates it generates
# are frozen to data/backtranslit/ and everything else runs on Windows.
#
# Build (from the repo root):
#   docker build -t script-gap-indicxlit docker/indicxlit
# Run: see scripts/backtranslit_candidates.py

# Official Python 3.8 image, pinned by digest (linux/amd64) so the build is repeatable.
FROM python:3.8-slim@sha256:314bc2fb0714b7807bf5699c98f0c73817e579799f2d91567ab7e9510f5601a5

# pip 24.1+ rejects the old metadata of omegaconf 2.0.x, which fairseq 0.12.2 pins.
RUN pip install --no-cache-dir pip==24.0

# CPU-only PyTorch (the job is small; no GPU needed inside the container).
RUN pip install --no-cache-dir --timeout 120 --retries 10 torch==1.13.1+cpu torchaudio==0.13.1+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu

# fairseq and the dependencies IndicXlit really uses. numpy < 1.24 because
# fairseq 0.12.2 still uses np.float; bitarray 3.11.0 is the newest version with a
# ready-made Python 3.8 wheel (the slim image has no C compiler to build newer ones).
RUN pip install --no-cache-dir --timeout 120 --retries 10 numpy==1.23.5 bitarray==3.11.0 fairseq==0.12.2 \
        sacremoses pandas tqdm ujson pydload indic-nlp-library mock tensorboardX pyarrow

# IndicXlit itself without its declared dependencies: they include urduhack
# (Urdu only), which would pull in TensorFlow.
RUN pip install --no-cache-dir --no-deps ai4bharat-transliteration==1.1.3

# IndicXlit imports urduhack.normalize at module level but calls it only for Urdu;
# the real urduhack imports TensorFlow, so a small stub (urduhack/__init__.py) stands in.
COPY urduhack /usr/local/lib/python3.8/site-packages/urduhack

WORKDIR /work
