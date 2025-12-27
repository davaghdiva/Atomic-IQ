# Atomic-IQ: Covariance Estimation and Mean-Variance Backtests

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18069453.svg)](https://doi.org/10.5281/zenodo.18069453)

This repository contains the reference implementation of the **atomic-IQ (AIQ)** covariance parametrization and its evaluation in a mean-variance (Max-Sharpe) backtest against a range of benchmark covariance estimators.

The entire pipeline can be executed with a single command:

```bash
python run_atomic_iq.py
```

This runs the parameter search, out-of-sample backtests, performance summary, ranking, and Sharpe ratio tests in sequence.

---

## Dependencies

- Python 3.10 or later

Python packages:

- numpy
- pandas
- scipy
- scikit-learn
- numba
- tqdm
- matplotlib
- optuna (needed for multi-trial parameter search in `aiq_mvo.py`)

A typical setup would install these via:

```bash
pip install -r requirements.txt
```

with `requirements.txt` containing at least the packages above.

---

## Data

The code expects two CSV files in the project root (or alternative paths specified in `defaults.py`):

- **Prices**: `./prices_multi_asset_master.csv`  
  - Must contain a `Date` column, parsable as dates.  
  - Remaining columns are asset prices. The first `n_assets` (see `DEFAULTS["n_assets"]`) are used.

- **Risk-free rate**: `./DGS3MO_monthly_rf.csv`  
  - Interpreted according to the settings in `defaults.py` (for example, annualized or effective monthly rate, and how it is aligned with the backtest dates).

If you prefer different filenames or locations, update the corresponding entries in `defaults.py`.

---

## Configuration

All central settings are collected in:

- `defaults.py`

Key items include:

- asset universe size (`n_assets`)
- lookback window length
- sample start and end dates for Parts 1 and 2
- transaction and optimization costs (basis points)
- AIQ parameter space, squeezing options, and Optuna trial count
- paths for prices and risk-free data
- scaling and normalization choices for covariance estimation

Adjust this file to match the configuration used in your empirical work.

---

## Main scripts

The project is organized around the following scripts:

- `run_atomic_iq.py`  
  Convenience runner that executes all stages in order:
  1. `aiq_mvo.py`
  2. `gs_sre_optuna.py`
  3. `diq_mvo.py`
  4. `diq_mvo_performance.py`
  5. `ranking_performance.py`
  6. `aiq_sr_tests.py`

- `aiq_cov.py`  
  Atomic-IQ covariance builders (AIQ1 and AIQ2): implements the IQ-based covariance parametrization and related helper functions.

- `aiq_mvo.py`  
  Part 1: AIQ parameter search and in-sample Max-Sharpe evaluation.  
  Uses the training window to select preferred AIQ1/AIQ2 parameter combinations (logits, squeezing levels, and related hyperparameters).

- `gs_sre_optuna.py`  
  Part 1: GS/SRE parameter search and in-sample Max-Sharpe evaluation.  
  Uses the training window to select preferred GS/SRE parameter combinations (threshold for GS and alpha for SRE).

- `diq_mvo.py`  
  Part 2: out-of-sample Max-Sharpe backtest.  
  Runs rolling mean-variance optimization using:
  - AIQ1 and AIQ2, and
  - benchmark estimators (Gerber variants, Ledoit-Wolf shrinkage, RMT-based estimators, and sample covariance).  
  Writes out-of-sample portfolio values, returns, and turnover to `OOS_results/`.

- `diq_mvo_optimizer.py`  
  Part 2: MVO optimizer and covariance back-ends.  
  - Computes asset moments from returns.  
  - Wraps different covariance estimators.  
  - Solves the Max-Sharpe optimization problems under weight and box constraints.

- `diq_mvo_trans_cost.py`  
  Part 2: Transaction-cost model used by the backtest to compute after-cost portfolio values for each rebalancing date.

- `diq_mvo_performance.py`  
  Part 3: performance evaluation.  
  - Reads out-of-sample portfolio value series from `OOS_results/`.  
  - Reconstructs after-cost excess returns.  
  - Computes performance measures (returns, volatility, Sharpe, Sortino, drawdowns, VaR, turnover, etc.).  
  - Exports performance tables (CSV and LaTeX).

- `ranking_performance.py`  
  Part 4: ranking and aggregation.  
  - Takes the performance table produced by `diq_mvo_performance.py`.  
  - Ranks all covariance methods across metrics.  
  - Produces an aggregate ranking and writes results to `ranking_results/`.

- `aiq_sr_tests.py`  
  Part 5: Sharpe ratio testing for Atomic IQ.  
  - Loads after-cost excess returns from the out-of-sample results.  
  - Treats AIQ1 (and optionally AIQ2) as benchmarks.  
  - Conducts hypothesis tests on Sharpe ratio differences, both full-sample and rolling window, using appropriate standard errors and bootstrap procedures.  
  - Writes results to `SR_test_results/`.

---

## Running the project

Once dependencies and data are in place and `defaults.py` is set as desired, running the entire study is:

```bash
python run_atomic_iq.py
```

This will:

1. Search and select AIQ/GS/SRE parameters on the training window.
2. Run the out-of-sample Max-Sharpe backtest for all covariance methods.
3. Compute and export performance statistics.
4. Rank methods across metrics.
5. Perform Sharpe ratio inference with AIQ as the benchmark.

---

## Outputs

The main output directories are:

- `OOS_results/`  
  - Out-of-sample portfolio value series and returns by method.  
  - Turnover and transaction costs.  
  - Performance summary files used by the ranking and Sharpe ratio scripts.

- `ranking_results/`  
  - Per-metric and aggregate rankings of all covariance estimators.  
  - Tables suitable for inclusion in the paper.

- `SR_test_results/`  
  - Full-sample and rolling Sharpe ratio estimates.  
  - Test statistics, confidence intervals, and p-values for AIQ versus competitors.

The exact file naming is documented in the headers of the corresponding scripts; these outputs are intended to align directly with the tables and figures in the paper.

---

## Reproducibility

To reproduce the results reported in the associated paper:

1. Use the same price and risk-free data as described in the empirical section.
2. Ensure that `defaults.py` matches the paper configuration (dates, asset universe, costs, scaling choices, AIQ parameter space, and Optuna settings). It currently does.
3. Run:

   ```bash
   python run_atomic_iq.py
   ```

4. Compare the performance, ranking, and Sharpe test outputs with the reported tables.

---

## Citing

If you use **Atomic-IQ** in academic work, please cite the software DOI (preferred). If you use the method described in our associated paper, please also cite the paper.

### Software (preferred)

**DOI:** https://doi.org/10.5281/zenodo.18069453

**BibTeX**
```bibtex
@software{smyth_abukhalaf_atomic_iq_2025,
  author  = {Smyth, William and Abu Khalaf, Layla},
  title   = {Atomic-IQ},
  year    = {2025},
  version = {0.1.0},
  doi     = {10.5281/zenodo.18069453},
  url     = {https://github.com/davaghdiva/Atomic-IQ}
}
```
**Paper**
```bibtex
@article{abukhalaf_smyth_squeezed_covariance_2025,
  author  = {Abu Khalaf, Layla and Smyth, William},
  title   = {Squeezed Covariance Matrix Estimation: Analytic Eigenvalue Control},
  journal = {SSRN Electronic Journal},
  year    = {2025},
  doi     = {10.2139/ssrn.XXXXXXX},
  url     = {https://ssrn.com/abstract=XXXXXXX},
  note    = {SSRN working paper}
}




