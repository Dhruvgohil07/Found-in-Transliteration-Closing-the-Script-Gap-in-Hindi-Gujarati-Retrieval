"""Loading experiment configs from YAML files in configs/ (SPECS §8)."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

# src/utils/config.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Every experiment config must name itself and fix its seed (CLAUDE.md §2.2).
REQUIRED_KEYS = ("experiment", "seed")


def load_config(
    path: str | Path, required: Iterable[str] = REQUIRED_KEYS
) -> dict[str, Any]:
    """Read a YAML config into a dict and check that the required keys exist.

    Raises ``TypeError`` if the file is not a YAML mapping and ``ValueError``
    if a required top-level key is missing.
    """
    path = resolve_path(path)
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError(f"{path}: config must be a YAML mapping, got {type(config).__name__}")

    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    return config


def resolve_path(path: str | Path) -> Path:
    """Turn a repo-relative path from a config into an absolute path.

    Configs store paths relative to the repo root (no local paths like
    ``D:\\...``), so the same config works on every machine. Absolute paths
    are returned unchanged.
    """
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_env_file(path: str | Path = ".env") -> list[str]:
    """Copy ``NAME=value`` lines from a git-ignored ``.env`` file into ``os.environ``.

    This is how secrets such as ``GEMINI_API_KEY`` reach the code without being
    written into a config or the repo. Blank lines and ``#`` comments are skipped,
    and a variable that is already set in the environment is never overwritten.
    A missing file is not an error. Returns the names that were set (never the values).
    """
    import os

    path = resolve_path(path)
    if not path.exists():
        return []
    loaded = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip().strip("\"'")
        if not sep or not name:
            raise ValueError(f"{path.name}:{n}: expected NAME=value")
        if name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded
