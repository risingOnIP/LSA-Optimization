"""
Run only after run_all.py has generated the CSVs.
"""

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
def require(file):
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

tfidf = stage1.loc[stage1.weight == "tfidf"].iloc[0]

print("\n=== Headline metrics for TF–IDF @ k=100 ===")
print(f"VAR_VAL  (variance %)        : {tfidf['var']*100:.1f}")
print(f"ERR_VAL  (recon error %)     : {tfidf['err']*100:.1f}")
print(f"ACC_VAL  (1-NN accuracy %)   : {tfidf['acc']*100:.1f}")

# ------------------------------------------------------------------
acc0  = noise_df.loc[noise_df.eps == 0,   "acc"].iloc[0]
acc03 = noise_df.loc[noise_df.eps == 0.3, "acc"].iloc[0]
drop  = (acc0 - acc03) * 100

print(f"\nDROP_VAL (accuracy drop at ε=0.3) : {drop:.1f}")

compression = comp_df.iloc[0, 0]
print(f"COMPRESSION_VAL (memory saved %)  : {compression:.1f}")


# other values

var_pct  = tfidf["var"] * 100
acc_pct  = tfidf["acc"] * 100
err_pct  = tfidf["err"] * 100

acc0   = noise_df.loc[noise_df.eps == 0.0, "acc"].iloc[0]
acc30  = noise_df.loc[noise_df.eps == 0.3, "acc"].iloc[0]
drop30 = (acc0 - acc30) * 100      # absolute percentage points

# --- pretty print -----------------------------------------------------------
print(f"VARIANCE_PERCENT  -> {var_pct:.1f}")
print(f"ACCURACY_PERCENT  -> {acc_pct:.1f}")
print(f"ERROR_PERCENT     -> {err_pct:.1f}")
print(f"DROP_AT_0p3_eps   -> {drop30:.1f}  # accuracy drop at ε=0.3")

print("\nText example:")
print(f"Tf-idf at $k = 100$ achieves the strongest performance, "
      f"capturing {var_pct:.1f}\\% variance, reaching "
      f"{acc_pct:.1f}\\% accuracy, and keeping reconstruction error to "
      f"{err_pct:.1f}\\%. "
      f"Accuracy declines by only {drop30:.1f}\\% when 30\\% of tokens "
      "are removed.")

compression = comp_df.iloc[0, 0]
print(f"\nCOMPRESSION_PERCENT = {compression:.1f}")
