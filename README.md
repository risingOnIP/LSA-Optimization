<p align="center"><img src="docs/assets/banner.svg" alt="LSA Optimization" width="100%"></p>

# LSA Optimization

**A reproducible study of term weighting, rank selection, and noise robustness in Latent Semantic Analysis.**

![Status](https://img.shields.io/badge/Status-Research-15803D?style=flat-square&labelColor=111827) ![Python](https://img.shields.io/badge/Python-3.8%2B-2563EB?style=flat-square&labelColor=111827&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-SVD-2563EB?style=flat-square&labelColor=111827&logo=scikitlearn&logoColor=white) ![UMAP](https://img.shields.io/badge/UMAP-Visualization-2563EB?style=flat-square&labelColor=111827) ![License](https://img.shields.io/badge/License-MIT-6B7280?style=flat-square&labelColor=111827)

This project measures how three design choices in Latent Semantic Analysis (LSA) affect retrieval quality on the 20 Newsgroups corpus. The best configuration is log-entropy weighting at rank k = 50, which reaches 48.0% 1-nearest-neighbor (1-NN) accuracy, and accuracy declines gradually rather than collapsing as tokens are removed.

## Overview

LSA represents documents by applying a truncated singular value decomposition (SVD) to a weighted term-document matrix. This repository runs a three-stage experiment on all 18,846 documents of 20 Newsgroups (20 labels, with headers, footers, and quoted replies removed) and writes every result to CSV files and figures with a single command.

- **Stage 1, term weighting:** compares raw counts, TF-IDF, and log-entropy at rank k = 100.
- **Stage 2, rank selection:** sweeps k over 10, 25, 50, 100, and 200 for the best weighting (log-entropy).
- **Stage 3, noise robustness:** randomly drops 0% to 50% of tokens at k = 100 and measures how accuracy degrades.
- **Reporting:** `reading.py` reads the CSVs and prints the best configuration (log-entropy at k = 50) and a LaTeX-ready summary sentence.

## Business Case

### The Problem

Teams with large document collections (support tickets, legal discovery sets, research archives, internal wikis) need semantic search and automatic categorization. Transformer embeddings deliver this but cost GPU time and money at scale. LSA is cheap, explainable, and runs on a CPU, yet practitioners usually choose its weighting scheme and rank by habit rather than by evidence.

### Who It Serves

- **Data scientists:** choose a lightweight text representation with measured trade-offs instead of defaults.
- **Search and knowledge-management engineers:** size a semantic index by picking the smallest rank that keeps accuracy.
- **Educators and students:** study a complete, reproducible example of linear algebra applied to NLP.

### The Outcome

The study produces evidence-based defaults: use log-entropy weighting at rank k = 50. On a 20-class benchmark that configuration reaches 48.0% 1-NN accuracy with an embedding 57.2% smaller than the non-zero values of the raw count matrix. Under token dropout, log-entropy LSA degrades gradually rather than failing abruptly. Every number regenerates with one command.

## How It Works

The pipeline turns raw text into weighted matrices, factors them with truncated SVD, and scores each configuration.

```mermaid
%%{init: {"theme": "neutral"}}%%
flowchart LR
    corpus["20 Newsgroups Corpus"] -->|"load"| preprocess["Preprocess"]
    preprocess -->|"count matrix"| weighting["Term Weighting"]
    weighting -->|"weighted matrix"| svd["Truncated SVD"]
    dropout["Token Dropout"] -->|"perturb"| svd
    svd -->|"embeddings"| metrics["Metrics"]
    svd -->|"project"| figures["Figures"]
    metrics -->|"plot"| figures
    metrics -->|"write"| csv["CSV Results"]
    csv -->|"summarize"| report["Reading Script"]
    class preprocess,weighting,svd,metrics,figures,csv,report core
    class dropout threat
    class corpus external
    classDef core fill:#2563EB,stroke:#1E40AF,color:#FFFFFF
    classDef guard fill:#15803D,stroke:#166534,color:#FFFFFF
    classDef threat fill:#B45309,stroke:#92400E,color:#FFFFFF
    classDef infra fill:#374151,stroke:#111827,color:#FFFFFF
    classDef external fill:#F3F4F6,stroke:#6B7280,color:#111827
```

1. **Load:** `src/data_loader.py` downloads 20 Newsgroups through scikit-learn and caches it in `data/`.
2. **Tokenize:** `src/preprocess.py` lowercases text, removes English stop words, and keeps terms found in at least 5 documents and at most 90% of documents.
3. **Weight:** `src/weighting.py` defines the three weightings by name: Raw counts, TF-IDF (L2-normalized), and Log-entropy (row-normalized).
4. **Factor:** `src/lsa.py` fits a randomized `TruncatedSVD` with seed 42.
5. **Score:** `src/metrics.py` computes variance explained, relative reconstruction error, and 1-NN accuracy on a seeded 80/20 split.
6. **Perturb:** `src/noise.py` zeroes a random fraction of non-zero entries to simulate token loss.
7. **Orchestrate:** `src/experiments.py` runs the three stages, `src/viz.py` draws the figures, and `run_all.py` computes compression at the best rank.
8. **Report:** `reading.py` picks the best weighting from `metrics_stage1.csv` and the best rank from `metrics_ranks.csv`, then prints the headline numbers.

## Results

All values below are copied from the committed result files, rounded to one decimal place.

**Stage 1: term weighting at k = 100** (`metrics_stage1.csv`)

| Weighting | Variance explained | Reconstruction error | 1-NN accuracy |
|---|---|---|---|
| Raw counts | 99.5% | 7.1% | 30.7% |
| TF-IDF | 11.2% | 93.7% | 44.5% |
| Log-entropy | 10.4% | 94.1% | 47.6% |

**Stage 2: rank sweep with log-entropy weighting** (`metrics_ranks.csv`)

| Rank k | 10 | 25 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| 1-NN accuracy | 32.0% | 40.9% | 48.0% | 47.6% | 47.0% |

**Stage 3: token dropout with log-entropy weighting at k = 100** (`metrics_noise.csv`)

| Token dropout rate | 0% | 10% | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|---|
| 1-NN accuracy | 47.6% | 45.6% | 43.1% | 39.5% | 35.6% | 32.5% |

**Best configuration: log-entropy at k = 50** (`metrics_ranks.csv`, `compression.csv`)

| Metric | Value |
|---|---|
| 1-NN accuracy | 48.0% |
| Variance explained | 6.9% |
| Reconstruction error | 95.9% |
| Compression | 57.2% |

Compression compares the float32 size of the 18,846 x 50 log-entropy embedding with the byte size of the int64 non-zero values array of the raw count matrix; the sparse index arrays are not counted.

Key findings:

- **Variance explained is a misleading quality signal.** Raw counts explain 99.5% of variance yet give the lowest accuracy (30.7%), because a few high-frequency terms dominate the decomposition.
- **Log-entropy is the best weighting.** It beats TF-IDF by 3.1 percentage points of 1-NN accuracy at k = 100.
- **k = 50 is the best rank.** Accuracy peaks at k = 50 (48.0%) and is slightly lower at k = 100 (47.6%) and k = 200 (47.0%).
- **Degradation under token dropout is gradual.** At k = 100, removing 30% of tokens costs 8.1 percentage points, with no sudden cliff.

The log-entropy embedding at k = 100 forms visible topic clusters in two dimensions, even though the 20 labels overlap heavily in the center.

![UMAP projection of log-entropy LSA embeddings at k = 100, colored by newsgroup](figs/fig1_umap_log_entropy.png)

Accuracy rises steeply up to k = 50 and then flattens, which makes k = 50 the best rank.

![1-NN accuracy versus SVD rank k for log-entropy weighting](figs/fig2_acc.png)

Accuracy falls steadily as the token dropout rate increases from 0% to 50%, with no abrupt collapse.

![1-NN accuracy versus token dropout rate for log-entropy weighting at k = 100](figs/fig3_noise.png)

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data | scikit-learn `fetch_20newsgroups` | Downloads and caches the 20 Newsgroups corpus |
| Data | pandas | Holds the corpus and writes result CSVs |
| Model | scikit-learn `CountVectorizer`, `TfidfTransformer` | Bag-of-words construction and TF-IDF weighting |
| Model | NumPy and SciPy sparse matrices | Log-entropy weighting and token dropout |
| Model | scikit-learn `TruncatedSVD`, `KNeighborsClassifier` | LSA factorization and 1-NN evaluation |
| Tooling | UMAP and Matplotlib | 2-D projections and result plots (`src/viz.py`) |

## Getting Started

### Prerequisites

- Python 3.8 or later.
- About 14 MB of network download on the first run (the 20 Newsgroups archive).
- 16 GB of RAM for the full pipeline (the reconstruction error metric works with dense 18,846 x 24,392 matrices).

### Installation

1. Clone the repository.

   ```bash
   git clone https://github.com/risingOnIP/lsa-optimization.git
   cd lsa-optimization
   ```

2. Create a virtual environment and install the dependencies.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

### Usage

Run the full experiment from the repository root.

```bash
python run_all.py
```

The script downloads the corpus into `data/`, runs all three stages, and writes `metrics_stage1.csv`, `metrics_ranks.csv`, `metrics_noise.csv`, `compression.csv`, and the PNG files in `figs/`. The UMAP projections take several minutes.

Print the result tables, the best configuration (log-entropy at k = 50), and a LaTeX-ready summary sentence.

```bash
python reading.py
```

To change the experiment grid, edit `ks` (ranks) or `eps_list` (token dropout rates) in `run_all.py`, or `COMPARE_K` in `src/experiments.py`. To add a weighting scheme, implement it in `src/weighting.py` and add it to `WEIGHTINGS` under its display name.

## Project Structure

```text
lsa-optimization/
├── src/
│   ├── data_loader.py     # download and load 20 Newsgroups
│   ├── preprocess.py      # tokenizer and count matrix
│   ├── weighting.py       # Raw counts, TF-IDF, and Log-entropy
│   ├── lsa.py             # truncated SVD wrapper
│   ├── metrics.py         # variance, reconstruction error, 1-NN accuracy
│   ├── noise.py           # token dropout
│   ├── viz.py             # UMAP projections and result plots
│   └── experiments.py     # stage 1 to 3 experiment runners
├── figs/                  # result figures
├── run_all.py             # runs every stage, writes outputs
├── reading.py             # prints the best configuration
├── metrics_stage1.csv     # weighting comparison at k = 100
├── metrics_ranks.csv      # rank sweep for log-entropy
├── metrics_noise.csv      # token dropout results at k = 100
├── compression.csv        # embedding size reduction at k = 50
├── requirements.txt
└── LICENSE
```

## Notes

- Every run uses seed 42 for the SVD, the 80/20 train and test split, and token dropout, so repeated runs with the same library versions reproduce the committed CSVs.
- Result CSVs are written to the repository root, where `reading.py` reads them.

## Roadmap

- [ ] Add repeated splits and seeds and report the mean and spread of each metric.
- [ ] Add retrieval metrics (precision at k, mean average precision) alongside 1-NN accuracy.
- [ ] Compute reconstruction error without densifying the matrix to reduce peak memory.
- [ ] Pin exact dependency versions for byte-identical reproduction.
- [ ] Move result CSVs into a dedicated `results/` directory.

## Acknowledgments

- The [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/), collected by Ken Lang, is downloaded at run time through [scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) and is not redistributed in this repository.
- [UMAP](https://umap-learn.readthedocs.io/) by Leland McInnes, John Healy, and James Melville provides the 2-D projections.

## License

Released under the [MIT License](LICENSE).

## Author

Developed by [Karthik Kaligotla](https://github.com/risingOnIP), named inventor on U.S. Patents [12,445,488 B2](https://patents.google.com/patent/US12445488B2/) and [12,598,158 B2](https://patents.google.com/patent/US12598158B2/).
