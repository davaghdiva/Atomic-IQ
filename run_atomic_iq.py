"""
Name    : run_atomic_iq.py
Author  : William Smyth & Layla Abu Khalaf
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : convenience runner to execute the atomic-IQ project in sequence:

1. aiq_mvo.py
   Part 1: parameter search for AIQ1 and AIQ2.
   This script runs the in-sample optimization for the atomic IQ
   covariance parametrization, selecting the preferred parameter
   combinations (logits, squeezing levels, etc.) over the training
   window.

2. gs_sre_optuna.py
   Parameter search for GS and SRE.
   This script runs the in-sample optimization for GS and SRE
   covariance parametrization, selecting the preferred parameter 
   combinations (threshold(GS),alpha(SRE)) over training window.

3. diq_mvo.py
   Part 2: out-of-sample MVO backtest across covariance methods.
   Using the selected AIQ1/AIQ2 parameters and a common universe of
   assets, this script runs a rolling mean–variance optimization
   backtest for all covariance estimators (AIQ1, AIQ2, GS1–GS3,
   LS1–LS5, NLS6–NLS8, HC, CRE, SRE), generating OOS portfolio value
   paths and returns for the Max–Sharpe strategy.

4. diq_mvo_performance.py
   Performance summary of results.
   This script reads the OOS portfolio value series from Part 2,
   reconstructs after-cost excess returns, and computes full-sample
   performance metrics for each method (e.g. mean return, volatility,
   Sharpe, Sortino, Calmar, maximum drawdown, 95% VaR, turnover).
   The resulting tables correspond to the main performance results in
   the paper.

5. ranking_performance.py
   Aggregate ranking across metrics.
   Based on the performance tables from Part 3, this script ranks each
   method on each metric and computes an aggregate rank, providing an
   overall ordering of the covariance estimators.

6. aiq_sr_tests.py
   Sharpe ratio inference and robustness checks.
   Using the after-cost excess returns from Part 2, this script:
     (i) recomputes full-sample annualized Sharpe ratios;
    (ii) performs moving-block bootstrap tests of Sharpe differences
         between AIQ1 (and optionally AIQ2) and all other methods;
   (iii) constructs 36-month rolling Sharpe series on monthly and
         quarterly grids and tests the mean Sharpe differential using
         Newey–West HAC standard errors and a block bootstrap.
   The tests are formulated with AIQ1 as the benchmark and are used to
   assess whether any competitor can be shown to have an equal or higher
   Sharpe ratio than AIQ1 in full-sample or rolling-window terms.
"""

import os
import subprocess
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def run_step(script_name: str):
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Could not find {script_name} in {BASE_DIR}")
    print(f"\n[runner] Starting {script_name} ...\n")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")
    print(f"\n[runner] Completed {script_name}.\n")

def main():
    for script in ["aiq_mvo.py", "gs_sre_optuna.py", "diq_mvo.py", "diq_mvo_performance.py", "ranking_performance.py", "aiq_sr_tests.py"]:
        run_step(script)
    print("\n[runner] All stages completed successfully.")

if __name__ == "__main__":
    main()
