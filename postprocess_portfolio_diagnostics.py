"""
Name    : postprocess_portfolio_diagnostics.py
Author  : William Smyth & Layla Abu Khalaf
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : run diagnostics OOS outcomes.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ORDER = ['AIQ1','AIQ2','CRE','GS1','GS2','GS3','GS_opt','HC',
         'LS1','LS2','LS3','LS4','LS5','NLS6','NLS7','NLS8','SRE','SRE_opt']

def compute_portfolio_diagnostics(result_pickle_path: Path) -> pd.DataFrame:
    with open(result_pickle_path, 'rb') as f:
        results = pickle.load(f)

    rows = []
    for method in ORDER:
        recs = results[method]['maxSharpe']
        active = [r for r in recs if np.sum(np.asarray(r['weights'], dtype=float)) > 0]
        weights = np.array([np.asarray(r['weights'], dtype=float) for r in active], dtype=float)
        max_w = weights.max(axis=1)
        eff_n = 1.0 / np.sum(weights**2, axis=1)
        turnover = np.array([float(r['weightDelta']) for r in active], dtype=float)

        portvals = np.array([float(r['portValue']) for r in recs], dtype=float)
        trans = np.array([float(r['transCost']) for r in recs], dtype=float)

        prev_vals = portvals[:-1]
        trans_active = trans[1:]
        tc_drag_monthly = np.divide(
            trans_active,
            prev_vals,
            out=np.zeros_like(trans_active),
            where=prev_vals != 0
        )

        rows.append({
            'Method': method,
            'Months': len(active),
            'Avg Max Weight': max_w.mean(),
            'Median Max Weight': np.median(max_w),
            'Avg Effective N': eff_n.mean(),
            'Median Effective N': np.median(eff_n),
            'Mean Monthly Turnover': turnover.mean(),
            'Median Monthly Turnover': np.median(turnover),
            'Annualized Turnover': turnover.mean() * 12.0,
            'Cumulative TC ($)': trans.sum(),
            'Cumulative TC Drag (%)': tc_drag_monthly.sum() * 100.0,
            'Annualized TC Drag (%)': tc_drag_monthly.mean() * 12.0 * 100.0,
            'End Value ($)': portvals[-1],
        })

    return pd.DataFrame(rows)

if __name__ == '__main__':
    project_root = Path('.')
    result_pickle_path = project_root / 'OOS_results' / 'result.pickle'
    out_path = project_root / 'OOS_results' / 'portfolio_diagnostics.csv'
    df = compute_portfolio_diagnostics(result_pickle_path)
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path}')
