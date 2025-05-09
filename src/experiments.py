from pathlib import Path, PurePath
import pandas as pd
from . import data_loader, preprocess, weighting, lsa, metrics, viz, noise

figdir = Path("figs"); figdir.mkdir(exist_ok=True)

def stage1_weight_compare():
    df = data_loader.fetch_20ng(Path("data"))
    X, cv = preprocess.docs_to_bow(df.text.tolist())

    funcs = {"raw": weighting.tf_raw,
             "tfidf": weighting.tf_tfidf,
             "logE": weighting.tf_log_entropy}
    rows = []
    for name, wf in funcs.items():
        print(f"\nStage 1 – weighting {name} at k=100")
        Xw = wf(X)
        svd, Z = lsa.lsa(Xw, k=100)
        rows.append(dict(
            weight=name,
            var=metrics.variance_explained(svd),
            err=metrics.recon_error(Xw, svd),
            acc=metrics.knn_accuracy(Z, df.label.values)
        ))
        viz.umap_scatter(Z, df.label.values,
                         f"{name} (k=100)",
                         figdir/f"fig1_umap_{name}.png")

    table = pd.DataFrame(rows)
    table.to_csv("metrics_stage1.csv", index=False)
    print("[TABLE] Saved metrics_stage1.csv – comparison of all weightings.")

    best = table.sort_values("acc", ascending=False).weight.iloc[0]
    print(f"Best weighting by harmonic-mean rule: {best}")
    return X, df, best

def stage2_rank_sweep(Xw, labels, ks):
    print("\nStage 2 – rank sweep for best weighting")
    import numpy as np, pandas as pd
    rows=[]
    for k in ks:
        svd, Z = lsa.lsa(Xw, k=k)
        rows.append((k,
                     metrics.variance_explained(svd),
                     metrics.recon_error(Xw, svd),
                     metrics.knn_accuracy(Z, labels)))
    out = pd.DataFrame(rows, columns=["k","var","err","acc"])
    out.to_csv("metrics_ranks.csv", index=False)
    print("[TABLE] Saved metrics_ranks.csv – metrics across k grid.")
    return out

def stage3_noise(Xw, labels, k, eps_list):
    print("\nStage 3 – noise robustness test")
    from . import lsa
    svd, _ = lsa.lsa(Xw, k=k)
    rows = []
    for eps in eps_list:
        Zn = svd.transform(noise.dropout(Xw, eps))
        rows.append((eps, metrics.knn_accuracy(Zn, labels)))
    out = pd.DataFrame(rows, columns=["eps","acc"])
    out.to_csv("metrics_noise.csv", index=False)
    print("[TABLE] Saved metrics_noise.csv – accuracy across dropout rates.")
    return out
