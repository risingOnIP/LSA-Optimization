"""The three experiment stages: weighting comparison, rank sweep, and noise robustness.

Each stage writes its metrics to a CSV file in the working directory. Weightings
are identified by their canonical names in ``weighting.WEIGHTINGS``.
"""

from pathlib import Path
import pandas as pd
from . import data_loader, preprocess, weighting, lsa, metrics, viz, noise

# Rank used to compare weightings (Stage 1) and to test noise robustness (Stage 3)
COMPARE_K = 100

figdir = Path("figs"); figdir.mkdir(exist_ok=True)


def stage1_weight_compare():
    """Compare Raw counts, TF-IDF, and Log-entropy weighting at k = COMPARE_K.

    Saves ``metrics_stage1.csv`` and one UMAP figure per weighting. Returns the
    raw count matrix, the corpus DataFrame, and the canonical name of the
    weighting with the highest 1-NN accuracy.
    """
    df = data_loader.fetch_20ng(Path("data"))
    X, cv = preprocess.docs_to_bow(df.text.tolist())

    rows = []
    for name, wf in weighting.WEIGHTINGS.items():
        print(f"\nStage 1: {name} weighting at k = {COMPARE_K}")
        Xw = wf(X)
        svd, Z = lsa.lsa(Xw, k=COMPARE_K)
        rows.append(dict(
            weight=name,
            var=metrics.variance_explained(svd),
            err=metrics.recon_error(Xw, svd),
            acc=metrics.knn_accuracy(Z, df.label.values)
        ))
        viz.umap_scatter(Z, df.label.values, name, COMPARE_K,
                         figdir/f"fig1_umap_{weighting.file_slug(name)}.png")

    table = pd.DataFrame(rows)
    table.to_csv("metrics_stage1.csv", index=False)
    print("[TABLE] Saved metrics_stage1.csv: comparison of all weightings.")

    best = table.sort_values("acc", ascending=False).weight.iloc[0]
    print(f"Best weighting by 1-NN accuracy: {best}")
    return X, df, best


def stage2_rank_sweep(Xw, labels, ks, name):
    """Sweep the SVD rank over ``ks`` for weighting ``name`` and save ``metrics_ranks.csv``."""
    print(f"\nStage 2: rank sweep for {name} weighting")
    rows=[]
    for k in ks:
        svd, Z = lsa.lsa(Xw, k=k)
        rows.append((k,
                     metrics.variance_explained(svd),
                     metrics.recon_error(Xw, svd),
                     metrics.knn_accuracy(Z, labels)))
    out = pd.DataFrame(rows, columns=["k","var","err","acc"])
    out.to_csv("metrics_ranks.csv", index=False)
    print("[TABLE] Saved metrics_ranks.csv: metrics across the rank grid.")
    best_k = int(out.loc[out["acc"].idxmax(), "k"])
    print(f"Best rank by 1-NN accuracy: k = {best_k}")
    return out


def stage3_noise(Xw, labels, k, eps_list, name):
    """Fit LSA on clean data, then measure accuracy under token dropout.

    Saves ``metrics_noise.csv`` with one row per dropout rate in ``eps_list``.
    """
    print(f"\nStage 3: token dropout test for {name} weighting at k = {k}")
    svd, _ = lsa.lsa(Xw, k=k)
    rows = []
    for eps in eps_list:
        Zn = svd.transform(noise.dropout(Xw, eps))
        rows.append((eps, metrics.knn_accuracy(Zn, labels)))
    out = pd.DataFrame(rows, columns=["eps","acc"])
    out.to_csv("metrics_noise.csv", index=False)
    print("[TABLE] Saved metrics_noise.csv: accuracy across token dropout rates.")
    return out
