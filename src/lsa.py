from sklearn.decomposition import TruncatedSVD
def lsa(X, k, seed=42):
    svd = TruncatedSVD(n_components=k,
                       algorithm="randomized",
                       random_state=seed)
    Z = svd.fit_transform(X)
    return svd, Z
