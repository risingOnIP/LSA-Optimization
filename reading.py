"""Print the headline numbers and a LaTeX-ready summary sentence from the result CSVs.

Run after run_all.py has written the CSVs. The best configuration combines the
weighting with the highest 1-NN accuracy in Stage 1 (metrics_stage1.csv) and the
rank k with the highest 1-NN accuracy in Stage 2 (metrics_ranks.csv).
"""

import pandas as pd
from pathlib import Path

# Rank used by Stage 1 and Stage 3 in run_all.py (experiments.COMPARE_K)
COMPARE_K = 100


def require(file):
    """Load a result CSV, or explain how to generate it if it is missing."""
    if not Path(file).exists():
        raise FileNotFoundError(
            f"{file} not found. Run `python run_all.py` first.")
    return pd.read_csv(file)


def main():
    """Print the result tables, the best configuration, and the summary sentence."""
    stage1   = require("metrics_stage1.csv")   # weighting comparison
    rank_df  = require("metrics_ranks.csv")    # sweep over k
    noise_df = require("metrics_noise.csv")    # token dropout robustness
    comp_df  = require("compression.csv")      # compression %

    print(f"\n=== Stage 1: term weighting at k = {COMPARE_K} (metrics_stage1.csv) ===")
    print(stage1.to_string(index=False))

    ranked = stage1.sort_values("acc", ascending=False)
    best, runner_up = ranked.iloc[0], ranked.iloc[1]
    name = best["weight"]
    margin = (best["acc"] - runner_up["acc"]) * 100

    print(f"\n=== Stage 2: rank sweep for {name} (metrics_ranks.csv) ===")
    print(rank_df.to_string(index=False))

    top = rank_df.loc[rank_df["acc"].idxmax()]
    k = int(top["k"])
    var_pct, err_pct, acc_pct = top["var"] * 100, top["err"] * 100, top["acc"] * 100

    acc0   = noise_df.loc[noise_df.eps == 0.0, "acc"].iloc[0]
    acc30  = noise_df.loc[noise_df.eps == 0.3, "acc"].iloc[0]
    drop30 = (acc0 - acc30) * 100      # absolute percentage points

    compression = comp_df.iloc[0, 0]

    print(f"\n=== Best configuration: {name} at k = {k} ===")
    headline = [
        ("Variance explained (%)", f"{var_pct:.1f}"),
        ("Reconstruction error (%)", f"{err_pct:.1f}"),
        ("1-NN accuracy (%)", f"{acc_pct:.1f}"),
        (f"Compression at k = {k} (%)", f"{compression:.1f}"),
        (f"Margin over {runner_up['weight']} at k = {COMPARE_K} (points)", f"{margin:.1f}"),
        (f"Drop at 30% token dropout, k = {COMPARE_K} (points)", f"{drop30:.1f}"),
    ]
    width = max(len(label) for label, _ in headline)
    for label, value in headline:
        print(f"{label.ljust(width)} : {value}")

    print("\nText example:")
    print(f"{name} weighting at $k = {k}$ achieves the strongest performance, "
          f"reaching {acc_pct:.1f}\\% 1-NN accuracy with an embedding "
          f"{compression:.1f}\\% smaller than the non-zero values of the raw "
          f"count matrix. At $k = {COMPARE_K}$, accuracy declines by only "
          f"{drop30:.1f} percentage points when 30\\% of tokens are removed.")


if __name__ == "__main__":
    main()
