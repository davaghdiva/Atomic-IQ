# Atomic-IQ: Constructive Covariance Estimation and Max-Sharpe Backtests

This repository contains the reference implementation of **Atomic-IQ (AIQ)** and the full empirical pipeline used for the long-only tangency (Max-Sharpe) study in the accompanying paper.

The project can be executed end to end with a single command:

```bash
python run_atomic_iq.py
```

That runner executes, in sequence:

1. AIQ parameter search (`aiq_mvo.py`)
2. GS/SRE parameter search (`gs_sre_optuna.py`)
3. Out-of-sample backtest across all methods (`diq_mvo.py`)
4. Performance summary generation (`diq_mvo_performance.py`)
5. Aggregate ranking generation (`ranking_performance.py`)
6. Sharpe-ratio difference summaries (`aiq_sr_tests.py`)
7. Portfolio implementation diagnostics (`postprocess_portfolio_diagnostics.py`)

## Repository contents

Core source files:

- `aiq_cov.py` — Atomic-IQ covariance builders (AIQ1 and AIQ2)
- `aiq_mvo.py` — in-sample AIQ parameter search
- `gs_sre_optuna.py` — in-sample GS/SRE parameter search
- `diq_mvo.py` — out-of-sample Max-Sharpe backtest
- `diq_mvo_optimizer.py` — optimization and covariance back-end wrappers
- `diq_mvo_trans_cost.py` — transaction-cost model
- `diq_mvo_performance.py` — performance tables and exports
- `ranking_performance.py` — per-metric and aggregate rankings
- `aiq_sr_tests.py` — full-sample and rolling Sharpe-difference summaries
- `postprocess_portfolio_diagnostics.py` — implementation diagnostics from saved portfolio paths
- `defaults.py` — central configuration file
- `run_atomic_iq.py` — convenience runner

Data files:

- `prices_multi_asset_master.csv`
- `DGS3MO_monthly_rf.csv`

Metadata and citation files:

- `LICENSE`
- `CITATION.cff`
- `CITATION.bib`
- `AI_DISCLOSURE.md`

Example outputs:

- `RESULTS.zip` — archived example outputs from a completed run of the project

## Requirements

- Python 3.10 or later

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The current requirements file includes:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `numba`
- `tqdm`
- `matplotlib`
- `optuna`

## Data expectations

The code expects the following CSV files in the project root unless alternative paths are set in `defaults.py`:

- `prices_multi_asset_master.csv`
  - must contain a `Date` column parsable as dates
  - remaining columns are asset prices
  - the first `DEFAULTS["n_assets"]` columns are used

- `DGS3MO_monthly_rf.csv`
  - interpreted according to the risk-free-rate settings in `defaults.py`

## Configuration

All central settings are defined in `defaults.py`, including:

- number of assets
- lookback window length
- in-sample and out-of-sample dates
- transaction-cost settings
- Atomic-IQ parameter space and Optuna trial count
- price and risk-free-rate paths
- scaling, weighting, and PSD-repair options

To reproduce the paper's baseline setup, use the provided defaults unchanged.

## Outputs

A full run creates the following output folders in the repository root:

- `OOS_results/`
  - portfolio value series
  - returns and turnover by method
  - `performance.csv` / `performance.tex`
  - `portfolio_diagnostics.csv`
  - `result.pickle`

- `ranking_results/`
  - per-metric ranks
  - aggregate rankings

- `SR_test_results/`
  - full-sample Sharpe summaries
  - rolling Sharpe-difference summaries

If you prefer not to generate these inside the repository root, adjust paths in the code before running.

## Reproducibility

To reproduce the study:

1. Ensure the input CSV files are present.
2. Check `defaults.py`.
3. Run:

   ```bash
   python run_atomic_iq.py
   ```

4. Compare the exported results with the paper tables and supplementary material.

## Citation

If you use this software, please cite the software record in `CITATION.cff` / `CITATION.bib`.

## Contact

- William Smyth
- Layla Abu Khalaf
- Contact: `drwss.academy@gmail.com`
