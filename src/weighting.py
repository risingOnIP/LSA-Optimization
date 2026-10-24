"""Term weighting schemes applied to a sparse document-term count matrix."""

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer


def tf_raw(X):
    """Return an unweighted copy of the raw count matrix."""
    return X.copy()


def tf_tfidf(X):
    """Apply TF-IDF weighting with L2 row normalization."""
    tfidf = TfidfTransformer(norm="l2", use_idf=True)
    return tfidf.fit_transform(X)


def tf_log_entropy(X):
    """Apply log-entropy weighting with L2 row normalization.

    The local weight is ``log(1 + count)``. The global weight of each term is
    ``1 + sum_d p log p / log(n_docs)``, where ``p`` is the share of the
    term's total count that falls in document ``d``.
    """
    df = np.array(X.sum(axis=0)).ravel()
    p = X.multiply(1/df)                 # P(d|t)
    logp = p.copy();  logp.data = np.log(logp.data+1e-10)
    g = 1 + (p.multiply(logp)).sum(axis=0) / np.log(X.shape[0])
    g = np.asarray(g).ravel()

    Xw = X.copy();  Xw.data = np.log1p(Xw.data)
    Xw = Xw.multiply(g)

    row_norm = np.sqrt(Xw.multiply(Xw).sum(1))
    row_norm[row_norm == 0] = 1
    Xw = Xw.multiply(1/row_norm)
    return Xw
