"""Reproduce condition-number and inactivity summaries quoted in the manuscript."""
from pathlib import Path
import csv,sys
import numpy as np
import pandas as pd
REPO_ROOT=Path.cwd(); sys.path.insert(0,str(REPO_ROOT))
from atomic_iq_sg.engine import load_prices,load_selected,covariance_for_method,_corr_from_cov,_kappa
from atomic_iq_sg.estimator import _state_scale
from atomic_iq_sg.config import LOOKBACK,EVALUATION_START,EVALUATION_END

def inactive_count(X,params):
    A=X.to_numpy(float); s=_state_scale(A,'mad'); a=np.abs(A/s)
    body=(a>params['c'])&(a<=params['delta']); tail=a>params['delta']
    diag=params['p']*body.sum(axis=0)+(1-params['p'])*tail.sum(axis=0)
    return int(np.sum(diag<=1e-12))

def main():
    selected=load_selected(REPO_ROOT/'RESULTS'/'calibration'/'selected_parameters.json'); params=selected['Atomic-IQ']['parameters']
    px=load_prices(REPO_ROOT/'data'/'prices_multi_asset_master.csv'); rets=px.pct_change(fill_method=None).dropna(); dates=rets.loc[EVALUATION_START:EVALUATION_END].index
    kappas={'AIQ':[],'HC':[]}; iw=ia=0
    for d in dates:
        t=rets.index.get_loc(d); X=rets.iloc[t-LOOKBACK:t]; Saiq,_=covariance_for_method('AIQ',X,selected); Shc=X.cov().to_numpy(float)
        for name,S in [('AIQ',Saiq),('HC',Shc)]: C,_=_corr_from_cov(S); kappas[name].append(_kappa(C))
        n=inactive_count(X,params); ia+=n; iw+=int(n>0)
    rows=[]
    for name in ['AIQ','HC']:
        v=np.asarray(kappas[name]); rows.append({'series':f'{name}_correlation_kappa_all_300','n_months':len(v),'mean':float(v.mean()),'min':float(v.min()),'max':float(v.max())})
    monthly=pd.read_csv(REPO_ROOT/'RESULTS'/'primary_gmv'/'monthly_gmv_results.csv'); idk=monthly[monthly.method.eq('ID_KAPPA')]
    alpha=idk.correlation_identity_shrinkage_alpha.to_numpy(float); matched=idk.matched_correlation_kappa.to_numpy(float); active=alpha>0
    rows += [
      {'series':'ID_KAPPA_matched_kappa_hybrid_all_300','n_months':len(matched),'mean':float(matched.mean()),'min':float(matched.min()),'max':float(matched.max())},
      {'series':'AIQ_correlation_kappa_when_IDK_active','n_months':int(active.sum()),'mean':float(np.asarray(kappas['AIQ'])[active].mean()),'min':float(np.asarray(kappas['AIQ'])[active].min()),'max':float(np.asarray(kappas['AIQ'])[active].max())},
      {'series':'ID_KAPPA_alpha_all_300','n_months':len(alpha),'mean':float(alpha.mean()),'min':float(alpha.min()),'max':float(alpha.max())}]
    with open(REPO_ROOT/'conditioning_summary.csv','w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=['series','n_months','mean','min','max']); w.writeheader(); w.writerows(rows)
    print(f'ID-kappa positive-shrinkage months={active.sum()}, zero-shrinkage months={(~active).sum()}')
    print(f'AIQ inactive windows={iw}, inactive asset-window count={ia}')

if __name__=='__main__': main()
