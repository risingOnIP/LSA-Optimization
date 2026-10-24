"""Figures: UMAP projections of document embeddings and result line plots.

Every title names the weighting with its canonical name from
``weighting.WEIGHTINGS`` and the SVD rank as ``k = <value>``.
"""

import umap, matplotlib.pyplot as plt
from pathlib import Path


def umap_scatter(Z, labels, weighting: str, k: int, fn: Path):
    """Project ``Z`` to 2-D with UMAP and save a scatter plot colored by label."""
    title = f"{weighting} LSA embedding (k = {k})"
    emb = umap.UMAP(n_components=2, random_state=42).fit_transform(Z)
    plt.figure(figsize=(6,5))
    plt.scatter(emb[:,0], emb[:,1], c=labels, s=6, cmap="tab20")
    plt.title(title); plt.tight_layout()
    plt.savefig(fn, dpi=300); plt.close()
    print(f"[FIGURE] Saved {fn}: UMAP projection of the {title}.")


def rank_plot(rank_df, column: str, ylabel: str, weighting: str, fn: Path):
    """Plot one Stage 2 metric against the rank k and save the figure."""
    title = f"{weighting}: {ylabel} by rank k"
    plt.figure(); plt.plot(rank_df["k"], rank_df[column], "o-")
    plt.xlabel("Rank k"); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(fn, dpi=300); plt.close()
    print(f"[FIGURE] Saved {fn}: {title}.")


def noise_plot(noise_df, weighting: str, k: int, fn: Path):
    """Plot 1-NN accuracy against the token dropout rate and save the figure."""
    title = f"{weighting} (k = {k}): 1-NN accuracy under token dropout"
    plt.figure(); plt.plot(noise_df["eps"], noise_df["acc"], "o-")
    plt.xlabel("Token dropout rate"); plt.ylabel("1-NN accuracy"); plt.title(title)
    plt.tight_layout(); plt.savefig(fn, dpi=300); plt.close()
    print(f"[FIGURE] Saved {fn}: {title}.")
