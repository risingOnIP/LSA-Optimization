"""Latent Semantic Analysis via truncated SVD."""

from sklearn.decomposition import TruncatedSVD


def lsa(X, k, seed=42):
    """Fit a rank-``k`` randomized truncated SVD.

    Returns the fitted model and the ``n_docs x k`` document embedding.
    """
    svd = TruncatedSVD(n_components=k,
                       algorithm="randomized",
                       random_state=seed)
    Z = svd.fit_transform(X)
    return svd, Z
