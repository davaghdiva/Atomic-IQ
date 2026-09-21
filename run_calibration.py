#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
from atomic_iq_sg.calibration import run_calibration

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'RESULTS'/'calibration'
sel=run_calibration(ROOT/'data'/'prices_multi_asset_master.csv',OUT)

a=pd.read_csv(OUT/'atomic_calibration_surface.csv')[['rank','c','delta','p','chi','net_vol_pct','conditioned_windows','min_rank']].head(12)
g=pd.read_csv(OUT/'gs_threshold_calibration.csv')[['rank','threshold','net_vol_pct']].head(12)
s=pd.read_csv(OUT/'sre_alpha_calibration.csv')[['rank','alpha','net_vol_pct']].head(12)
report=f'''# Calibration results

Parameters are selected using realised long-only GMV performance from September 1989 through December 1999, using a 20-month rolling window, drifted pre-trade turnover and 10 bp transaction costs. `chi=0.90` is fixed structurally for the primary Atomic-IQ estimator; the calibration searches `c`, `delta` and `p`.

## Frozen selections

```json
{json.dumps(sel,indent=2)}
```

## Atomic-IQ: leading calibration points

{a.to_markdown(index=False,floatfmt='.6f')}

## Gerber threshold: leading calibration points

{g.to_markdown(index=False,floatfmt='.6f')}

## SRE alpha: leading calibration points

{s.to_markdown(index=False,floatfmt='.6f')}
'''
(OUT/'CALIBRATION_RESULTS.md').write_text(report)
print('\nCALIBRATION SELECTIONS\n')
for k,v in sel.items():
    if k!='calibration_period': print(k,':',v)
