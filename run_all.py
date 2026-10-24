"""Run all three experiment stages and write the CSV results and figures.

Usage: python run_all.py
"""

import pandas as pd
from pathlib import Path
from src import experiments, weighting, lsa, viz

FIGS = Path("figs")


def main():
    """Run stages 1 to 3, save figures, and compute compression at the best rank."""
    # Stage 1: compare weightings at k = 100 and pick the best by 1-NN accuracy
    X, df, best = experiments.stage1_weight_compare()
    X_best = weighting.WEIGHTINGS[best](X)

    # Stage 2: rank sweep for the best weighting
    ks = [10,25,50,100,200]
    rank_df = experiments.stage2_rank_sweep(X_best, df.label.values, ks, best)
    best_k = int(rank_df.loc[rank_df["acc"].idxmax(), "k"])

    viz.rank_plot(rank_df, "var", "Variance explained", best, FIGS/"fig2_var.png")
    viz.rank_plot(rank_df, "err", "Reconstruction error", best, FIGS/"fig2_err.png")
    viz.rank_plot(rank_df, "acc", "1-NN accuracy", best, FIGS/"fig2_acc.png")

    # Stage 3: token dropout robustness at the Stage 1 rank
    noise_df = experiments.stage3_noise(X_best, df.label.values,
                                        k=experiments.COMPARE_K,
                                        eps_list=[0,0.1,0.2,0.3,0.4,0.5],
                                        name=best)
    viz.noise_plot(noise_df, best, experiments.COMPARE_K, FIGS/"fig3_noise.png")

    # Compression: float32 embedding at the best rank vs the non-zero values of the raw count matrix
    bytes_full = X.data.nbytes
    svd_best, Z_best = lsa.lsa(X_best, best_k)
    bytes_Z = Z_best.astype("float32").nbytes
    compression = 100 * (1 - bytes_Z/bytes_full)
    print(f"[INFO] Compression of the {best} embedding at k = {best_k}: {compression:.1f}%")
    pd.DataFrame({"compression_percent":[compression]}).to_csv("compression.csv",index=False)


if __name__ == "__main__":
    main()
