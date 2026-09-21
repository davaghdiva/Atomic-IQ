from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import estimator, benchmarks
from .config import LOOKBACK,COST_RATE,EVALUATION_START,EVALUATION_END,TARGET_MIN_EIG,BOOTSTRAP_REPS,BOOTSTRAP_BLOCK,BOOTSTRAP_SEED


def load_prices(path):
    px=pd.read_csv(path,parse_dates=['Date'],dayfirst=True).set_index('Date').sort_index()
    return px.resample('ME',label='right').last()


def load_rf(path,index):
    rf=pd.read_csv(path,index_col=0,parse_dates=True).iloc[:,0]
    rf.index=pd.to_datetime(rf.index)
    union=rf.index.union(index).sort_values()
    return rf.reindex(union).ffill().reindex(index)


def corr_condition_cov(S,target=TARGET_MIN_EIG):
    S=np.asarray(S,float); S=0.5*(S+S.T)
    sd=np.sqrt(np.clip(np.diag(S),1e-18,None))
    C=S/np.outer(sd,sd); C=0.5*(C+C.T); np.fill_diagonal(C,1.0)
    ev=np.linalg.eigvalsh(C); lam=float(ev.min()); rank=int(np.sum(ev>1e-8))
    alpha=0.0
    if lam<target:
        alpha=float(np.clip((target-lam)/(1-lam),0,1))
        C=(1-alpha)*C+alpha*np.eye(len(C)); S=np.outer(sd,sd)*C
    return S,alpha,lam,rank


def solve_gmv(S):
    S=np.asarray(S,float); n=S.shape[0]; x0=np.ones(n)/n
    out=minimize(lambda w:float(w@S@w),x0,jac=lambda w:2*S@w,
                 bounds=[(0,1)]*n,
                 constraints=[{'type':'eq','fun':lambda w:float(w.sum()-1),'jac':lambda w:np.ones(n)}],
                 method='SLSQP',options={'ftol':1e-12,'maxiter':100})
    if not out.success: raise RuntimeError(out.message)
    w=np.asarray(out.x,float); w[np.abs(w)<1e-4]=0
    return w/w.sum()


def _drift(w,asset_r):
    v=w*(1+asset_r); return v/v.sum()


def _kappa(S):
    ev=np.linalg.eigvalsh(0.5*(S+S.T)); lo=max(float(ev[0]),1e-18); return float(ev[-1]/lo)


def _corr_from_cov(S):
    S=np.asarray(S,float); S=0.5*(S+S.T)
    sd=np.sqrt(np.clip(np.diag(S),1e-18,None))
    C=S/np.outer(sd,sd); C=0.5*(C+C.T); np.fill_diagonal(C,1.0)
    return C,sd


def correlation_identity_shrink_to_kappa(sample_cov,target_corr_kappa):
    C,sd=_corr_from_cov(sample_cov)
    ev=np.linalg.eigvalsh(C); lmin=float(ev[0]); lmax=float(ev[-1])
    if lmin<=0:
        C=C+(1e-12-lmin)*np.eye(len(C));
        d=np.sqrt(np.clip(np.diag(C),1e-18,None)); C=C/np.outer(d,d); np.fill_diagonal(C,1.0)
        ev=np.linalg.eigvalsh(C); lmin=float(ev[0]); lmax=float(ev[-1])
    sample_k=lmax/lmin
    k=float(target_corr_kappa)
    if k>=sample_k or k<=1:
        a=0.0
    else:
        den=(lmax-1.0)+k*(1.0-lmin)
        a=float(np.clip((lmax-k*lmin)/den,0,1)) if abs(den)>1e-18 else 1.0
    C1=(1-a)*C+a*np.eye(len(C))
    out=np.outer(sd,sd)*C1
    return out,a,_kappa(C1)


def load_selected(path):
    with open(path) as f: return json.load(f)


def covariance_for_method(name,df,selected,primary_cov=None):
    X=df.to_numpy(float)
    if name=='AIQ':
        p=selected['Atomic-IQ']['parameters']
        return estimator.covariance(X,state_scale='mad',return_diagnostics=True,**p)
    if name=='HC': S=df.cov().to_numpy()
    elif name=='LS1': S=np.asarray(benchmarks.cov1Para(df))
    elif name=='LS2': S=np.asarray(benchmarks.cov2Para(df))
    elif name=='LS3': S=np.asarray(benchmarks.covCor(df))
    elif name=='LS4': S=np.asarray(benchmarks.covDiag(df))
    elif name=='LS5': S=np.asarray(benchmarks.covMarket(df))
    elif name=='NLS6': S=np.asarray(benchmarks.GIS(df))
    elif name=='NLS7': S=np.asarray(benchmarks.LIS(df))
    elif name=='NLS8': S=np.asarray(benchmarks.QIS(df))
    elif name=='CRE': S=np.asarray(benchmarks.RMT1(X)[0])
    elif name=='SRE_STAR':
        a=float(selected['SRE_STAR']['alpha']); S=np.asarray(benchmarks.RMT2(X,alpha=a)[0])
    elif name=='GS_STAR':
        th=float(selected['GS_STAR']['threshold']); S=np.asarray(benchmarks.gerber_cov_stat(X,th)[0])
    elif name=='ID_KAPPA':
        if primary_cov is None: raise ValueError('primary_cov required for ID_KAPPA')
        S0=df.cov().to_numpy()
        Cp,_=_corr_from_cov(primary_cov)
        S,a,k=correlation_identity_shrink_to_kappa(S0,_kappa(Cp))
        Cmatch,_=_corr_from_cov(S)
        return S,{'conditioning_alpha':0.0,'pre_lambda_min':float(np.linalg.eigvalsh(Cmatch).min()),
                  'rank_pre_conditioning':int(np.linalg.matrix_rank(Cmatch,tol=1e-8)),
                  'correlation_identity_shrinkage_alpha':a,'matched_correlation_kappa':k}
    else: raise KeyError(name)
    S,alpha,lam,rank=corr_condition_cov(S)
    return S,{'conditioning_alpha':alpha,'pre_lambda_min':lam,'rank_pre_conditioning':rank}


COV_METHODS=['AIQ','HC','LS1','LS2','LS3','LS4','LS5','NLS6','NLS7','NLS8','CRE','SRE_STAR','GS_STAR','ID_KAPPA']
FULL_METHODS=COV_METHODS+['EW']


def run_gmv(prices_csv,rf_csv,selected_json,methods=FULL_METHODS,start=EVALUATION_START,end=EVALUATION_END):
    selected=load_selected(selected_json)
    px=load_prices(prices_csv); rets=px.pct_change(fill_method=None).dropna()
    eval_dates=rets.loc[start:end].index; rf=load_rf(rf_csv,eval_dates)
    states={m:{'w':None,'prev_asset_r':None} for m in methods}; rows=[]
    for d in eval_dates:
        t=rets.index.get_loc(d); X=rets.iloc[t-LOOKBACK:t]; now=rets.iloc[t].to_numpy(float)
        primary_cov,primary_diag=covariance_for_method('AIQ',X,selected)
        for m in methods:
            if m=='EW':
                w=np.ones(X.shape[1])/X.shape[1]
                diag={'conditioning_alpha':0.0,'pre_lambda_min':np.nan,'rank_pre_conditioning':X.shape[1]}
            else:
                if m=='AIQ': S,diag=primary_cov,primary_diag
                else: S,diag=covariance_for_method(m,X,selected,primary_cov=primary_cov)
                w=solve_gmv(S)
            st=states[m]
            pre=np.zeros(len(w)) if st['w'] is None else _drift(st['w'],st['prev_asset_r'])
            turnover=float(np.abs(w-pre).sum()); gross=float(w@now)
            net=float((1-COST_RATE*turnover)*(1+gross)-1)
            row={'Date':d,'method':m,'gross_return':gross,'net_return':net,'rf':float(rf.loc[d]),
                 'gross_turnover':turnover,'conditioning_alpha':float(diag.get('conditioning_alpha',0)),
                 'pre_lambda_min':float(diag.get('pre_lambda_min',np.nan)),
                 'rank_pre_conditioning':int(diag.get('rank_pre_conditioning',len(w)))}
            if 'correlation_identity_shrinkage_alpha' in diag:
                row['correlation_identity_shrinkage_alpha']=diag['correlation_identity_shrinkage_alpha']
            if 'matched_correlation_kappa' in diag:
                row['matched_correlation_kappa']=diag['matched_correlation_kappa']
            rows.append(row); st['w']=w; st['prev_asset_r']=now.copy()
    return pd.DataFrame(rows)


def _expected_shortfall_loss_pct(r, tail_prob=0.05):
    r=np.asarray(r,float); n=len(r)
    k=max(1,int(math.ceil(tail_prob*n)))
    worst=np.sort(r)[:k]
    return float(-np.mean(worst)*100)


def reported_metrics(g):
    r=g.net_return.to_numpy(float); rf=g.rf.to_numpy(float)
    wealth_net=np.cumprod(1+r)
    dd=wealth_net/np.maximum.accumulate(wealth_net)-1
    downside=np.minimum(r,0.0)
    out={
        'n_months':len(g),
        'first_date':str(pd.to_datetime(g.Date).min().date()),
        'last_date':str(pd.to_datetime(g.Date).max().date()),
        'realised_vol_10bp_pct':float(np.std(r,ddof=1)*math.sqrt(12)*100),
        'sharpe_10bp':float(np.mean(r-rf)/np.std(r,ddof=1)*math.sqrt(12)),
        'max_drawdown_10bp_pct':float(-dd.min()*100),
        'es95_10bp_pct':_expected_shortfall_loss_pct(r,0.05),
        'downside_deviation_10bp_pct':float(math.sqrt(12*np.mean(downside**2))*100),
        'terminal_wealth_10bp_from_100':float(100*wealth_net[-1]),
    }
    return out


def diagnostic_metrics(g):
    out={
        'n_months':len(g),
        'first_date':str(pd.to_datetime(g.Date).min().date()),
        'last_date':str(pd.to_datetime(g.Date).max().date()),
        'conditioned_windows':int((g.conditioning_alpha>0).sum()),
        'min_rank':int(g.rank_pre_conditioning.min()),
    }
    if 'correlation_identity_shrinkage_alpha' in g and g.correlation_identity_shrinkage_alpha.notna().any():
        out['mean_correlation_identity_shrinkage_alpha']=float(g.correlation_identity_shrinkage_alpha.mean())
    return out


def summary_table(long_df):
    return pd.DataFrame([{'method':m,**reported_metrics(g)} for m,g in long_df.groupby('method',sort=False)]).sort_values('realised_vol_10bp_pct').reset_index(drop=True)


def ranking_tables(summary):
    z=summary.copy()
    risk=[
        ('realised_vol_10bp_pct','vol_rank',True),
        ('max_drawdown_10bp_pct','max_drawdown_rank',True),
        ('es95_10bp_pct','es95_rank',True),
        ('downside_deviation_10bp_pct','downside_deviation_rank',True),
    ]
    for col,name,ascending in risk:
        z[name]=z[col].rank(method='min',ascending=ascending).astype(int)
    z['sharpe_rank']=z['sharpe_10bp'].rank(method='min',ascending=False).astype(int)
    z['terminal_wealth_10bp_rank']=z['terminal_wealth_10bp_from_100'].rank(method='min',ascending=False).astype(int)
    risk_rank_cols=[x[1] for x in risk]
    z['risk_aggregate']=z[risk_rank_cols].sum(axis=1)
    z['risk_overall_rank']=z['risk_aggregate'].rank(method='min',ascending=True).astype(int)
    z['broad_aggregate']=z[risk_rank_cols+['sharpe_rank','terminal_wealth_10bp_rank']].sum(axis=1)
    z['broad_overall_rank']=z['broad_aggregate'].rank(method='min',ascending=True).astype(int)
    risk_order=z.sort_values(['risk_aggregate','realised_vol_10bp_pct','method']).index
    broad_order=z.sort_values(['broad_aggregate','realised_vol_10bp_pct','method']).index
    risk_table=z.loc[risk_order,['risk_overall_rank','method']+risk_rank_cols+['risk_aggregate']].reset_index(drop=True)
    broad_table=z.loc[broad_order,['broad_overall_rank','method']+risk_rank_cols+['sharpe_rank','terminal_wealth_10bp_rank','broad_aggregate']].reset_index(drop=True)
    return risk_table,broad_table


def diagnostics_table(long_df):
    return pd.DataFrame([{'method':m,**diagnostic_metrics(g)} for m,g in long_df.groupby('method',sort=False)])


def subperiod_table(long_df):
    periods=[('2000-2004','2000-01-01','2004-12-31'),('2005-2009','2005-01-01','2009-12-31'),('2010-2014','2010-01-01','2014-12-31'),('2015-2019','2015-01-01','2019-12-31'),('2020-2024','2020-01-01','2024-12-31')]
    z=long_df.copy(); z.Date=pd.to_datetime(z.Date); rows=[]
    for label,a,b in periods:
        q=z[(z.Date>=a)&(z.Date<=b)]
        for m,g in q.groupby('method'): rows.append({'period':label,'method':m,**reported_metrics(g)})
    return pd.DataFrame(rows)

def moving_block_diff(a,b,reps=BOOTSTRAP_REPS,block=BOOTSTRAP_BLOCK,seed=BOOTSTRAP_SEED):
    a=np.asarray(a,float); b=np.asarray(b,float); n=len(a); rng=np.random.default_rng(seed)
    vals=np.empty(reps); av=lambda x:np.std(x,ddof=1)*math.sqrt(12)*100
    for k in range(reps):
        starts=rng.integers(0,n-block+1,math.ceil(n/block)); ix=np.concatenate([np.arange(s,s+block) for s in starts])[:n]
        vals[k]=av(a[ix])-av(b[ix])
    d=av(a)-av(b); return d,float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))


def primary_pairwise_inference(long_df):
    w=long_df.pivot(index='Date',columns='method',values='net_return'); rows=[]
    for b in [x for x in w.columns if x!='AIQ']:
        d,lo,hi=moving_block_diff(w.AIQ,w[b]); rows.append({'atomic':'AIQ','benchmark':b,'diff_ann_vol_pp':d,'ci95_lo_pp':lo,'ci95_hi_pp':hi,'ci_excludes_zero':bool(lo>0 or hi<0)})
    return pd.DataFrame(rows).sort_values('diff_ann_vol_pp')
