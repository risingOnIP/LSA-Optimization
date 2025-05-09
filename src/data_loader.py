from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from pathlib import Path

def fetch_20ng(local_dir: Path) -> pd.DataFrame:
    local_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading / loading 20 Newsgroups corpus …")
    data = fetch_20newsgroups(subset="all",
                              remove=("headers", "footers", "quotes"),
                              data_home=str(local_dir))
    print(f"Loaded {len(data.data)} documents with {len(data.target_names)} labels.")
    return pd.DataFrame({"text": data.data, "label": data.target})
