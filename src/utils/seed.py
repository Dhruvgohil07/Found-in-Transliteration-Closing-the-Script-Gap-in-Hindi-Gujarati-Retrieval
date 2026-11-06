"""Fixing random seeds so that runs are reproducible (SPECS §8)."""

import random

import numpy as np
import torch

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch (CPU and all GPUs) with the same value.

    With ``deterministic=True``, cuDNN is also told to use only deterministic
    algorithms. That makes GPU training bit-for-bit repeatable, but slower,
    so it is off by default and turned on per experiment in its config.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe to call when no GPU is present

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
