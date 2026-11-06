"""Environment smoke test: is the setup ready to run experiments?

Run from the repo root inside the conda env `script-gap`:

    python -m scripts.check_env

Each check prints PASS or FAIL with a short detail; the exit code is 1 if any
check fails. The checks are:

1. Python is 3.11.
2. Python runs in UTF-8 mode (needed to read Hindi with ir_datasets on Windows).
3. Installed package versions match the pins in requirements.txt.
4. Every library we use can be imported.
5. torch sees the GPU and can run an fp16 matrix multiply on it.
6. PyTerrier starts the JVM on Java 17 or newer.
7. A tiny Hindi BM25 index works end to end: Devanagari words are indexed and
   the right document comes first. Terrier's default English tokeniser drops
   every non-Latin character, so the index must use tokeniser="utf".
"""

import importlib
import math
import sys
import tempfile
from collections.abc import Callable
from importlib import metadata

from src.utils.config import REPO_ROOT

IMPORTS = [
    "torch", "sentence_transformers", "transformers", "ir_datasets", "pyterrier",
    "ir_measures", "ranx", "indic_transliteration", "pandas", "yaml", "matplotlib",
    "numpy", "scipy", "pytest",
]

# Toy Hindi documents written for this check (not from any dataset).
HINDI_DOCS = [
    {"docno": "d1", "text": "किसान आंदोलन दिल्ली में जारी है"},
    {"docno": "d2", "text": "भारत ने क्रिकेट मैच जीता"},
    {"docno": "d3", "text": "आज मौसम साफ़ है"},
]
HINDI_QUERY = "किसान आंदोलन"  # "farmers' protest": should rank d1 first
HINDI_UNIQUE_TERMS = 14  # distinct words in HINDI_DOCS ("है" appears twice)


def check_python() -> str:
    version = sys.version_info
    if (version.major, version.minor) != (3, 11):
        raise RuntimeError(f"expected Python 3.11, got {sys.version.split()[0]}")
    return sys.version.split()[0]


def check_utf8_mode() -> str:
    # ir_datasets reads some files with the default encoding (cp1252 on Windows),
    # which cannot decode Hindi; UTF-8 mode makes the default UTF-8.
    if not sys.flags.utf8_mode:
        raise RuntimeError("set PYTHONUTF8=1 in the environment, or run with 'python -X utf8'")
    return "on"


def check_pinned_versions() -> str:
    pins = {}
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==")
            pins[name.strip()] = version.strip()

    wrong = []
    for name, pinned in pins.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "missing"
        if installed != pinned:
            wrong.append(f"{name} {installed} (pinned {pinned})")
    if wrong:
        raise RuntimeError("; ".join(wrong))
    return f"{len(pins)} pinned packages match"


def check_imports() -> str:
    failed = []
    for module in IMPORTS:
        try:
            importlib.import_module(module)
        # Broad on purpose: a broken install can raise more than ImportError
        # (e.g. OSError when a torch DLL fails to load on Windows).
        except Exception as e:  # noqa: BLE001
            failed.append(f"{module} ({type(e).__name__}: {e})")
    if failed:
        raise RuntimeError("; ".join(failed))
    return f"{len(IMPORTS)} modules"


def check_gpu() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} cannot see a CUDA GPU")
    # A real computation on the GPU, not just "a device exists".
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    y = (x @ x).float().sum().item()
    if not math.isfinite(y):
        raise RuntimeError(f"fp16 matmul on the GPU returned {y}")
    props = torch.cuda.get_device_properties(0)
    return (
        f"{props.name}, {props.total_memory / 1024**3:.1f} GB, "
        f"CUDA {torch.version.cuda}, fp16 matmul OK"
    )


def check_java() -> str:
    import pyterrier as pt

    if not pt.java.started():
        pt.java.init()
    java_version = pt.java.J.System.getProperty("java.version")
    major = int(java_version.split(".")[0])
    if major < 17:
        raise RuntimeError(f"Java {java_version} found, need 17 or newer")
    return f"Java {java_version}, Terrier {pt.terrier.version()}"


def check_hindi_bm25() -> str:
    import pyterrier as pt

    if not pt.java.started():
        pt.java.init()
    # ignore_cleanup_errors: on Windows the JVM can keep index files open for
    # a moment after close(), which would otherwise make cleanup fail.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as index_dir:
        ref = pt.terrier.IterDictIndexer(index_dir, overwrite=True, tokeniser="utf").index(
            HINDI_DOCS
        )
        index = pt.IndexFactory.of(ref)
        try:
            n_terms = index.getCollectionStatistics().getNumberOfUniqueTerms()
            if n_terms != HINDI_UNIQUE_TERMS:
                raise RuntimeError(
                    f"index has {n_terms} terms, expected {HINDI_UNIQUE_TERMS}"
                    " (is the tokeniser dropping Devanagari?)"
                )
            results = pt.terrier.Retriever(index, wmodel="BM25").search(HINDI_QUERY)
            if results.empty or results.iloc[0]["docno"] != "d1":
                raise RuntimeError(f"expected d1 first, got {results['docno'].tolist()}")
        finally:
            index.close()
    return f"{n_terms} Devanagari terms indexed, BM25 ranks the right doc first"


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("Python 3.11", check_python),
    ("UTF-8 mode", check_utf8_mode),
    ("Pinned versions", check_pinned_versions),
    ("Imports", check_imports),
    ("GPU (torch + CUDA)", check_gpu),
    ("Java + PyTerrier", check_java),
    ("Hindi BM25 end to end", check_hindi_bm25),
]


def main() -> int:
    failures = 0
    for name, check in CHECKS:
        try:
            detail = check()
            print(f"PASS  {name}: {detail}")
        # Broad on purpose: one failed check (Java errors included) must not
        # stop the others from running and reporting.
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
