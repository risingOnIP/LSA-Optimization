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

