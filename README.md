
## Installation

1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies** (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the full experiment pipeline:
```bash
python run_all.py
```
- This will download the dataset, run all experiments, and save results as CSVs and figures in the `figs/` directory.

### 2. Summarize results for reporting:
```bash
python reading.py
```
- Prints headline metrics and guidance for writing up results 

## Outputs

- **CSV files:** Contain all key metrics for each experiment stage.
- **Figures:** Plots of variance explained, reconstruction error, accuracy vs. SVD rank, and noise robustness.
- **Console output:** Summary tables and ready-to-paste text for reports.

## Customization

- **Change SVD ranks:** Edit the `ks` list in `run_all.py`.
- **Change noise levels:** Edit the `eps_list` in `run_all.py`.
- **Add new weighting schemes:** Implement in `src/weighting.py` and add to `experiments.py`.

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions.

