"""Load the 20 Newsgroups corpus into a DataFrame."""

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from pathlib import Path


def fetch_20ng(local_dir: Path) -> pd.DataFrame:
    """Download (or load from cache) all 20 Newsgroups documents.

    Headers, footers, and quoted replies are removed. Returns a DataFrame
    with a ``text`` column and an integer ``label`` column.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    print("Loading 20 Newsgroups corpus (downloaded on first run)...")
    data = fetch_20newsgroups(subset="all",
                              remove=("headers", "footers", "quotes"),
                              data_home=str(local_dir))
    print(f"Loaded {len(data.data)} documents with {len(data.target_names)} labels.")
    return pd.DataFrame({"text": data.data, "label": data.target})
