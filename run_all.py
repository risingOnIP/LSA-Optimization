"""Run all three experiment stages and write the CSV results and figures.

Usage: python run_all.py
"""

import matplotlib.pyplot as plt, pandas as pd
from src import experiments, weighting, lsa


def main():
    """Run stages 1 to 3, save figures, and compute the k = 50 compression figure."""
    # Stage 1: compare weightings and pick the best by 1-NN accuracy
    X, df, best = experiments.stage1_weight_compare()
    best_func = {"raw": weighting.tf_raw,
                 "tfidf": weighting.tf_tfidf,
                 "logE": weighting.tf_log_entropy}[best]
    X_best = best_func(X)

    # Stage 2: rank sweep
    ks = [10,25,50,100,200]
    rank_df = experiments.stage2_rank_sweep(X_best, df.label.values, ks)

    plt.figure(); plt.plot(rank_df["k"], rank_df["var"], "o-")
    plt.xlabel("k"); plt.ylabel("variance explained")
    plt.tight_layout(); fn="figs/fig2_var.png"; plt.savefig(fn,dpi=300); plt.close()
    print(f"[FIGURE] Saved {fn}: variance explained vs rank.")

    plt.figure(); plt.plot(rank_df["k"], rank_df["err"], "o-")
    plt.xlabel("k"); plt.ylabel("reconstruction error")
    plt.tight_layout(); fn="figs/fig2_err.png"; plt.savefig(fn,dpi=300); plt.close()
    print(f"[FIGURE] Saved {fn}: reconstruction error vs rank.")

    plt.figure(); plt.plot(rank_df["k"], rank_df["acc"], "o-")
    plt.xlabel("k"); plt.ylabel("1-NN accuracy")
    plt.tight_layout(); fn="figs/fig2_acc.png"; plt.savefig(fn,dpi=300); plt.close()
    print(f"[FIGURE] Saved {fn}: 1-NN accuracy vs rank.")

    # Stage 3: noise robustness
    noise_df = experiments.stage3_noise(X_best, df.label.values, k=100,
                                        eps_list=[0,0.1,0.2,0.3,0.4,0.5])

    plt.figure(); plt.plot(noise_df.eps, noise_df.acc, "o-")
    plt.xlabel("dropout ε"); plt.ylabel("1-NN accuracy")
    plt.tight_layout(); fn="figs/fig3_noise.png"; plt.savefig(fn, dpi=300); plt.close()
    print(f"[FIGURE] Saved {fn}: accuracy vs token-dropout.")

    # Compression: float32 k = 50 embedding vs the non-zero values of the raw count matrix
    bytes_full = X.data.nbytes
    svd50, Z50 = lsa.lsa(X_best, 50)
    bytes_Z = Z50.astype("float32").nbytes
    compression = 100 * (1 - bytes_Z/bytes_full)
    print(f"[INFO] Compression ratio at k=50 = {compression:.1f}%")
    pd.DataFrame({"compression_percent":[compression]}).to_csv("compression.csv",index=False)


if __name__ == "__main__":
    main()
