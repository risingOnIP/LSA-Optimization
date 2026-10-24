"""Two-dimensional UMAP visualizations of document embeddings."""

import umap, matplotlib.pyplot as plt
from pathlib import Path


def umap_scatter(Z, labels, title: str, fn: Path):
    """Project ``Z`` to 2-D with UMAP and save a scatter plot colored by label."""
    emb = umap.UMAP(n_components=2, random_state=42).fit_transform(Z)
    plt.figure(figsize=(6,5))
    plt.scatter(emb[:,0], emb[:,1], c=labels, s=6, cmap="tab20")
    plt.title(title); plt.tight_layout()
    plt.savefig(fn, dpi=300); plt.close()
    print(f"[FIGURE] Saved {fn.name}: 2-D UMAP scatter for {title}.")
