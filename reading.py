"""Print headline numbers and a LaTeX-ready summary sentence from the result CSVs.

Run only after run_all.py has generated the CSVs.
"""

import pandas as pd
from pathlib import Path

WEIGHT_NAMES = {"raw": "Raw counts", "tfidf": "TF-IDF", "logE": "Log-entropy"}


def require(file):
    """Load a result CSV, or explain how to generate it if it is missing."""
    if not Path(file).exists():
        raise FileNotFoundError(
            f"{file} not found.  Run `python run_all.py` first.")
    return pd.read_csv(file)


stage1   = require("metrics_stage1.csv")   # weighting comparison
rank_df  = require("metrics_ranks.csv")    # sweep over k
noise_df = require("metrics_noise.csv")    # noise robustness
comp_df  = require("compression.csv")      # compression %

# ------------------------------------------------------------------
print("\n=== Table 1 values (k = 100) ===")
print(stage1.to_string(index=False))

# Best weighting by 1-NN accuracy, the same rule run_all.py uses
best = stage1.sort_values("acc", ascending=False).iloc[0]
best_name = WEIGHT_NAMES.get(best["weight"], best["weight"])

var_pct = best["var"] * 100
err_pct = best["err"] * 100
acc_pct = best["acc"] * 100

print(f"\n=== Headline metrics for {best_name} @ k=100 ===")
print(f"VAR_VAL  (variance %)        : {var_pct:.1f}")
print(f"ERR_VAL  (recon error %)     : {err_pct:.1f}")
print(f"ACC_VAL  (1-NN accuracy %)   : {acc_pct:.1f}")

# ------------------------------------------------------------------
acc0   = noise_df.loc[noise_df.eps == 0.0, "acc"].iloc[0]
acc30  = noise_df.loc[noise_df.eps == 0.3, "acc"].iloc[0]
drop30 = (acc0 - acc30) * 100      # absolute percentage points

compression = comp_df.iloc[0, 0]

print(f"\nDROP_VAL (accuracy drop at ε=0.3, points) : {drop30:.1f}")
print(f"COMPRESSION_VAL (memory saved %)          : {compression:.1f}")

print("\nText example:")
print(f"{best_name} at $k = 100$ achieves the strongest performance, "
      f"capturing {var_pct:.1f}\\% variance, reaching "
      f"{acc_pct:.1f}\\% accuracy, and keeping reconstruction error to "
      f"{err_pct:.1f}\\%. "
      f"Accuracy declines by only {drop30:.1f} percentage points when 30\\% "
      "of tokens are removed.")
