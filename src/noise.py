import numpy as np
def dropout(X, eps, seed=42):
    rng = np.random.default_rng(seed)
    Xc = X.copy().tocsr()
    mask = rng.random(Xc.data.size) > eps
    Xc.data = Xc.data * mask
    Xc.eliminate_zeros()
    return Xc
