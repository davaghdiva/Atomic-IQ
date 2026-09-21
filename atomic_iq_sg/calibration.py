from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd

from .config import *
from .engine import load_prices,solve_gmv,corr_condition_cov
from . import benchmarks
from .estimator import MAD_NORMAL


def batch_active_gmv(S,tol=1e-10,maxit=30):
    """Exact active-set solver for batches of long-only fully-invested GMV problems."""
    S=np.asarray(S,float); m,n,_=S.shape; masks=np.full(m,(1<<n)-1,dtype=np.int32); W=np.zeros((m,n)); done=np.zeros(m,bool); seen=[set() for _ in range(m)]
    for _ in range(maxit):
        ids0=np.where(~done)[0]
        if not len(ids0): break
        for i in ids0:
            if int(masks[i]) in seen[i]: raise RuntimeError('active-set cycle')
            seen[i].add(int(masks[i]))
        for mask in np.unique(masks[ids0]):
            ids=ids0[masks[ids0]==mask]; A=np.array([j for j in range(n) if mask&(1<<j)],int)
            SA=S[ids][:,A][:,:,A]; one=np.ones((len(ids),len(A)))
            x=np.linalg.solve(SA,one[...,None])[...,0]; wa=x/x.sum(axis=1)[:,None]
            minpos=np.argmin(wa,axis=1); neg=wa[np.arange(len(ids)),minpos] < -tol
            for local,i in enumerate(ids[neg]): masks[i] &= ~(1<<int(A[minpos[neg][local]]))
            posids=ids[~neg]; posa=wa[~neg]
            if not len(posids): continue
            W[posids]=0; W[np.ix_(posids,A)]=np.maximum(posa,0); W[posids]/=W[posids].sum(axis=1)[:,None]
            g=np.einsum('mij,mj->mi',S[posids],W[posids],optimize=True); q=g[:,A].mean(axis=1)
            I=np.array([j for j in range(n) if not(mask&(1<<j))],int)
            if not len(I): done[posids]=True; continue
            diff=g[:,I]-q[:,None]; jmin=np.argmin(diff,axis=1); viol=diff[np.arange(len(posids)),jmin] < -tol
            done[posids[~viol]]=True
            for local,i in enumerate(posids[viol]): masks[i] |= (1<<int(I[jmin[viol][local]]))
    if not np.all(done): raise RuntimeError('active-set solves unresolved')
    return W


def _period(prices_csv):
    px=load_prices(prices_csv); rets=px.pct_change(fill_method=None).dropna(); R=rets.to_numpy(float); idx=rets.index
    tidx=[t for t in range(LOOKBACK,len(rets)) if idx[t]<=pd.Timestamp(CALIBRATION_END)]
    if len(tidx)!=124 or str(idx[tidx[0]].date())!='1989-09-30': raise RuntimeError('unexpected calibration chronology')
    return rets,R,idx,tidx


def _metrics(net,gross,turn):
    return float(np.std(net,ddof=1)*math.sqrt(12)*100),float(np.std(gross,ddof=1)*math.sqrt(12)*100),float(np.mean(turn)*12)


def calibrate_atomic(prices_csv):
    rets,R,idx,tidx=_period(prices_csv); pv=P_GRID; chi=PRIMARY_CHI; ew=chi*np.sqrt(pv*(1-pv)); rows=[]
    for c,d in itertools.product(C_GRID,DELTA_GRID):
        M=len(pv); prevW=None; prevR=None; net=np.empty((len(tidx),M)); gross=np.empty_like(net); turn=np.empty_like(net); cond=np.zeros(M,int); minrank=np.full(M,R.shape[1],int)
        for wi,t in enumerate(tidx):
            X=R[t-LOOKBACK:t]; med=np.median(X,axis=0); ss=np.clip(MAD_NORMAL*np.median(np.abs(X-med),axis=0),1e-12,None); lift=np.clip(np.std(X,axis=0,ddof=1),1e-12,None)
            Z=X/ss; a=np.abs(Z); sg=np.sign(Z); b=np.where((a>c)&(a<=d),sg,0.0); tau=np.where(a>d,sg,0.0)
            BB=b.T@b; TT=tau.T@tau; BT=b.T@tau
            G=pv[:,None,None]*BB+(1-pv)[:,None,None]*TT+ew[:,None,None]*(BT+BT.T)
            dg=np.diagonal(G,axis1=1,axis2=2).copy(); bad=dg<=1e-12; inv=1/np.sqrt(np.where(bad,1,dg)); C=G*inv[:,:,None]*inv[:,None,:]
            for k in range(M):
                ix=np.where(bad[k])[0]
                if len(ix): C[k,ix,:]=0; C[k,:,ix]=0
                C[k]=0.5*(C[k]+C[k].T); np.fill_diagonal(C[k],1)
            ev=np.linalg.eigvalsh(C); lam=ev[:,0]; rank=np.sum(ev>1e-8,axis=1); aa=np.zeros_like(lam); mask=lam<TARGET_MIN_EIG; aa[mask]=np.clip((TARGET_MIN_EIG-lam[mask])/(1-lam[mask]),0,1)
            C=(1-aa)[:,None,None]*C+aa[:,None,None]*np.eye(C.shape[1])[None,:,:]; cond+=aa>0; minrank=np.minimum(minrank,rank)
            S=C*lift[None,:,None]*lift[None,None,:]; W=batch_active_gmv(S)
            pre=np.zeros_like(W) if prevW is None else (prevW*(1+prevR[None,:]))/(prevW*(1+prevR[None,:])).sum(axis=1)[:,None]
            to=np.abs(W-pre).sum(axis=1); rg=W@R[t]; rn=(1-COST_RATE*to)*(1+rg)-1
            net[wi]=rn; gross[wi]=rg; turn[wi]=to; prevW=W; prevR=R[t].copy()
        for j,p in enumerate(pv):
            nv,gv,to=_metrics(net[:,j],gross[:,j],turn[:,j]); rows.append({'c':float(c),'delta':float(d),'p':float(p),'chi':chi,'net_vol_pct':nv,'gross_vol_pct':gv,'turnover_ann':to,'conditioned_windows':int(cond[j]),'min_rank':int(minrank[j])})
    df=pd.DataFrame(rows).sort_values(['net_vol_pct','c','delta','p']).reset_index(drop=True); df.insert(0,'rank',np.arange(1,len(df)+1))
    win=df.iloc[0].to_dict()
    exact=_evaluate_single_atomic(rets,R,idx,tidx,win)
    return df,win,exact


def _evaluate_single_atomic(rets,R,idx,tidx,p):
    from . import estimator
    prev=None; prevR=None; net=[]; gross=[]; turn=[]
    for t in tidx:
        X=R[t-LOOKBACK:t]; S=estimator.covariance(X,c=p['c'],delta=p['delta'],p=p['p'],chi=p['chi'],state_scale='mad'); w=solve_gmv(S); pre=np.zeros_like(w) if prev is None else (prev*(1+prevR))/(prev*(1+prevR)).sum(); to=np.abs(w-pre).sum(); rg=w@R[t]; rn=(1-COST_RATE*to)*(1+rg)-1; net.append(rn);gross.append(rg);turn.append(to);prev=w;prevR=R[t].copy()
    nv,gv,to=_metrics(np.array(net),np.array(gross),np.array(turn)); return {'net_vol_pct':nv,'gross_vol_pct':gv,'turnover_ann':to}


def _condition_batch(S):
    S=np.asarray(S,float); sd=np.sqrt(np.clip(np.diagonal(S,axis1=1,axis2=2),1e-18,None)); C=S/(sd[:,:,None]*sd[:,None,:]); C=0.5*(C+C.transpose(0,2,1));
    for k in range(len(C)): np.fill_diagonal(C[k],1)
    lam=np.linalg.eigvalsh(C)[:,0]; aa=np.zeros_like(lam); mask=lam<TARGET_MIN_EIG; aa[mask]=np.clip((TARGET_MIN_EIG-lam[mask])/(1-lam[mask]),0,1); C=(1-aa)[:,None,None]*C+aa[:,None,None]*np.eye(C.shape[1])[None,:,:]; return C*(sd[:,:,None]*sd[:,None,:])


def _eval_param_batch(prices_csv,params,cov_batch):
    rets,R,idx,tidx=_period(prices_csv); M=len(params); prevW=None; prevR=None; net=np.empty((len(tidx),M)); gross=np.empty_like(net); turn=np.empty_like(net)
    for wi,t in enumerate(tidx):
        S=_condition_batch(cov_batch(R[t-LOOKBACK:t],params)); W=batch_active_gmv(S); pre=np.zeros_like(W) if prevW is None else (prevW*(1+prevR[None,:]))/(prevW*(1+prevR[None,:])).sum(axis=1)[:,None]; to=np.abs(W-pre).sum(axis=1); rg=W@R[t]; rn=(1-COST_RATE*to)*(1+rg)-1; net[wi]=rn; gross[wi]=rg; turn[wi]=to; prevW=W; prevR=R[t].copy()
    return net,gross,turn


def _gs_batch(X,ths):
    n,p=X.shape; sd=X.std(axis=0); out=[]
    for th in ths:
        hi=X>=th*sd; lo=X<=-th*sd; small=np.abs(X)<th*sd; pos=(hi[:,:,None]&hi[:,None,:])|(lo[:,:,None]&lo[:,None,:]); neg=(hi[:,:,None]&lo[:,None,:])|(lo[:,:,None]&hi[:,None,:]); nn=small[:,:,None]&small[:,None,:]; num=pos.sum(axis=0)-neg.sum(axis=0); den=n-nn.sum(axis=0); C=np.divide(num,den,out=np.zeros_like(num,dtype=float),where=den!=0); out.append(np.outer(sd,sd)*C)
    return np.stack(out)


def calibrate_gs(prices_csv):
    th=GS_THRESHOLD_GRID; net,gross,turn=_eval_param_batch(prices_csv,th,_gs_batch); rows=[]
    for j,x in enumerate(th):
        nv,gv,to=_metrics(net[:,j],gross[:,j],turn[:,j]); rows.append({'threshold':float(x),'net_vol_pct':nv,'gross_vol_pct':gv,'turnover_ann':to})
    df=pd.DataFrame(rows).sort_values(['net_vol_pct','threshold']).reset_index(drop=True); df.insert(0,'rank',np.arange(1,len(df)+1)); return df,df.iloc[0].to_dict()


def _sre_batch(X,alphas):
    cov0=np.cov(X,rowvar=0); corr0=benchmarks.cov2corr(cov0); eVal0,eVec0=benchmarks.getPCA(corr0); eMax0,_=benchmarks.findMaxEval(np.diag(eVal0), X.shape[0]/X.shape[1], 0.01); nFacts0=eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0); eValL,eVecL=eVal0[:nFacts0,:nFacts0],eVec0[:,:nFacts0]; eValR,eVecR=eVal0[nFacts0:,nFacts0:],eVec0[:,nFacts0:]; A=eVecL@eValL@eVecL.T; Rr=eVecR@eValR@eVecR.T; D=np.diag(np.diag(Rr)); sd=np.sqrt(np.diag(cov0)); out=[]
    for a in alphas: out.append((A+a*Rr+(1-a)*D)*np.outer(sd,sd))
    return np.stack(out)


def calibrate_sre(prices_csv):
    al=SRE_ALPHA_GRID; net,gross,turn=_eval_param_batch(prices_csv,al,_sre_batch); rows=[]
    for j,x in enumerate(al):
        nv,gv,to=_metrics(net[:,j],gross[:,j],turn[:,j]); rows.append({'alpha':float(x),'net_vol_pct':nv,'gross_vol_pct':gv,'turnover_ann':to})
    df=pd.DataFrame(rows).sort_values(['net_vol_pct','alpha']).reset_index(drop=True); df.insert(0,'rank',np.arange(1,len(df)+1)); return df,df.iloc[0].to_dict()


def _evaluate_single_benchmark(prices_csv, kind, value):
    rets,R,idx,tidx=_period(prices_csv); prev=None; prevR=None; net=[]; gross=[]; turn=[]
    for t in tidx:
        X=rets.iloc[t-LOOKBACK:t]
        if kind=='GS': S=benchmarks.gerber_cov_stat(X.to_numpy(float),threshold=float(value))[0]
        elif kind=='SRE': S=benchmarks.RMT2(X.to_numpy(float),alpha=float(value))[0]
        else: raise KeyError(kind)
        S,_,_,_=corr_condition_cov(S); w=solve_gmv(S); pre=np.zeros_like(w) if prev is None else (prev*(1+prevR))/(prev*(1+prevR)).sum(); to=np.abs(w-pre).sum(); rg=w@R[t]; rn=(1-COST_RATE*to)*(1+rg)-1; net.append(rn); gross.append(rg); turn.append(to); prev=w; prevR=R[t].copy()
    nv,gv,to=_metrics(np.array(net),np.array(gross),np.array(turn)); return {'net_vol_pct':nv,'gross_vol_pct':gv,'turnover_ann':to}


def run_calibration(prices_csv,outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    adf,awin,aexact=calibrate_atomic(prices_csv); gdf,gwin=calibrate_gs(prices_csv); sdf,swin=calibrate_sre(prices_csv)
    gexact=_evaluate_single_benchmark(prices_csv,'GS',gwin['threshold']); sexact=_evaluate_single_benchmark(prices_csv,'SRE',swin['alpha'])
    adf[['rank','c','delta','p','chi','net_vol_pct','conditioned_windows','min_rank']].to_csv(outdir/'atomic_calibration_surface.csv',index=False)
    gdf[['rank','threshold','net_vol_pct']].to_csv(outdir/'gs_threshold_calibration.csv',index=False)
    sdf[['rank','alpha','net_vol_pct']].to_csv(outdir/'sre_alpha_calibration.csv',index=False)
    selected={'calibration_period':{'realised_start':'1989-09-30','realised_end':'1999-12-31','n_months':124,'lookback_months':LOOKBACK,'transaction_cost_bps':COST_BPS,'objective':'minimise after-cost annualised realised long-only GMV volatility'},
              'Atomic-IQ':{'parameters':{'c':float(awin['c']),'delta':float(awin['delta']),'p':float(awin['p']),'chi':float(awin['chi'])},'grid_rank':int(awin['rank']),'calibration_net_vol_pct':float(aexact['net_vol_pct']),'selection_note':'chi=0.90 fixed structurally; c, delta and p selected by calibration'},
              'GS_STAR':{'threshold':float(gwin['threshold']),'grid_rank':int(gwin['rank']),'calibration_net_vol_pct':float(gwin['net_vol_pct']),'selected_exact_slsqp_net_vol_pct':float(gexact['net_vol_pct'])},
              'SRE_STAR':{'alpha':float(swin['alpha']),'grid_rank':int(swin['rank']),'calibration_net_vol_pct':float(swin['net_vol_pct']),'selected_exact_slsqp_net_vol_pct':float(sexact['net_vol_pct'])}}
    (outdir/'selected_parameters.json').write_text(json.dumps(selected,indent=2))
    return selected
