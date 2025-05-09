import numpy as np, scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier

def variance_explained(svd):
    return svd.explained_variance_ratio_.sum()

def recon_error(X, svd):
    Xhat = svd.inverse_transform(svd.transform(X))
    if sp.issparse(X):
        num = np.linalg.norm((X - Xhat).data)
        denom = np.linalg.norm(X.data)
    else:
        num = np.linalg.norm(X - Xhat)
        denom = np.linalg.norm(X)
    return num/denom

def knn_accuracy(Z, labels, train_frac=0.8, seed=42):
    n = len(labels)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    split = int(train_frac*n)
    train, test = idx[:split], idx[split:]
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z[train], labels[train])
    return knn.score(Z[test], labels[test])
