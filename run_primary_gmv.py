#!/usr/bin/env python3
from pathlib import Path
from atomic_iq_sg.engine import (
    run_gmv, summary_table, diagnostics_table, subperiod_table,
    primary_pairwise_inference, ranking_tables
)

ROOT = Path(__file__).resolve().parent
CAL = ROOT / 'RESULTS' / 'calibration' / 'selected_parameters.json'
OUT = ROOT / 'RESULTS' / 'primary_gmv'
OUT.mkdir(parents=True, exist_ok=True)
if not CAL.exists():
    raise SystemExit('Run python run_calibration.py first.')

long = run_gmv(ROOT/'data'/'prices_multi_asset_master.csv',ROOT/'data'/'DGS3MO_monthly_rf.csv',CAL)
long.to_csv(OUT/'monthly_gmv_results.csv',index=False)
summary=summary_table(long); summary.to_csv(OUT/'gmv_summary.csv',index=False)
diag=diagnostics_table(long); diag.to_csv(OUT/'gmv_diagnostics.csv',index=False)
sub=subperiod_table(long); sub.to_csv(OUT/'gmv_subperiods.csv',index=False)
inf=primary_pairwise_inference(long); inf.to_csv(OUT/'paired_bootstrap_primary_vs_all.csv',index=False)
publication_cols=['method','realised_vol_10bp_pct','sharpe_10bp','max_drawdown_10bp_pct','es95_10bp_pct','downside_deviation_10bp_pct','terminal_wealth_10bp_from_100']
publication=summary[publication_cols]; publication.to_csv(OUT/'gmv_publication_table.csv',index=False)
risk_rank,broad_rank=ranking_tables(summary); risk_rank.to_csv(OUT/'gmv_risk_ranking.csv',index=False); broad_rank.to_csv(OUT/'gmv_broad_ranking.csv',index=False)

report=f"""# Primary GMV results

The parameters in `../calibration/selected_parameters.json` are held fixed throughout January 2000–December 2024.

## Full-period GMV outcomes

The table is ordered by annualised realised volatility, the GMV objective. Every reported portfolio outcome uses the 10 bp net-return path. Expected shortfall is the positive magnitude of the mean return in the worst 5% of realised months; downside deviation uses a zero-return target. Terminal wealth is a wealth index normalised to 100 at inception and includes 10 bp transaction costs.

{publication.to_markdown(index=False,floatfmt='.6f')}

Annualised return and VaR are not reported. Turnover is computed from drifted pre-trade weights and is used to charge the 10 bp transaction cost, but it is not included as a standalone reported outcome.

## Panel A — GMV risk ranking

{risk_rank.to_markdown(index=False)}

## Panel B — broad outcome ranking

{broad_rank.to_markdown(index=False)}

## Paired moving-block bootstrap: AIQ minus comparator annualised volatility

Negative values favour AIQ.

{inf.to_markdown(index=False,floatfmt='.6f')}

The complete five-year subperiod output is in `gmv_subperiods.csv`. PSD/conditioning diagnostics are retained separately in `gmv_diagnostics.csv`.
"""
(OUT/'PRIMARY_GMV_RESULTS.md').write_text(report)

print('\nPRIMARY GMV PUBLICATION TABLE\n'); print(publication.to_string(index=False))
print('\nPAIRED AIQ VOLATILITY DIFFERENCES\n'); print(inf.to_string(index=False))
