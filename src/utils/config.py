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
