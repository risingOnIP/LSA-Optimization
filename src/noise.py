"""Synthetic noise for robustness tests."""

import numpy as np


def dropout(X, eps, seed=42):
    """Randomly zero a fraction ``eps`` of the non-zero entries of ``X``."""
    rng = np.random.default_rng(seed)
    Xc = X.copy().tocsr()
    mask = rng.random(Xc.data.size) > eps
    Xc.data = Xc.data * mask
    Xc.eliminate_zeros()
    return Xc
