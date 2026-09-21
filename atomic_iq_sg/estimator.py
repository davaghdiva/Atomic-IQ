from __future__ import annotations

import math
import numpy as np

MAD_NORMAL = 1.482602218505602
TARGET_MIN_EIG = 1e-6


def kernel(p: float, chi: float) -> np.ndarray:
    p=float(p); chi=float(chi)
    if not 0 <= p <= 1: raise ValueError('p must lie in [0,1]')
    if not 0 <= chi <= 1: raise ValueError('chi must lie in [0,1]')
    off=chi*math.sqrt(p*(1-p))
    return np.array([[p,off],[off,1-p]],float)


def _state_scale(X: np.ndarray, method: str) -> np.ndarray:
    X=np.asarray(X,float)
    if method=='sd':
        s=np.std(X,axis=0,ddof=1)
    elif method=='mad':
        med=np.median(X,axis=0)
        s=MAD_NORMAL*np.median(np.abs(X-med),axis=0)
    elif method=='qn':
        from statsmodels.robust.scale import qn_scale
        s=np.array([qn_scale(X[:,j]) for j in range(X.shape[1])],float)
    else:
        raise ValueError("state_scale must be one of {'sd','mad','qn'}")
    return np.clip(s,1e-12,None)


def _normalise(G: np.ndarray) -> np.ndarray:
    G=0.5*(G+G.T)
    d=np.diag(G).copy()
    inactive=d<=1e-12
    inv=1/np.sqrt(np.where(inactive,1.0,d))
    C=(G*inv).T*inv
    if np.any(inactive):
        ix=np.where(inactive)[0]
        C[ix,:]=0; C[:,ix]=0
    C=0.5*(C+C.T)
    np.fill_diagonal(C,1.0)
    return C


def condition_correlation(C: np.ndarray, target_min_eig: float=TARGET_MIN_EIG):
    C=0.5*(np.asarray(C,float)+np.asarray(C,float).T)
    lam=float(np.linalg.eigvalsh(C).min())
    if lam>=target_min_eig:
        return C,0.0
    alpha=float(np.clip((target_min_eig-lam)/(1-lam),0,1))
    out=(1-alpha)*C+alpha*np.eye(C.shape[0])
    return out,alpha


def correlation(X, *, c: float, delta: float, p: float, chi: float,
                state_scale: str='mad', target_min_eig: float=TARGET_MIN_EIG,
                return_diagnostics: bool=False):
    X=np.asarray(X,float)
    if not 0 <= c < delta: raise ValueError('require 0 <= c < delta')
    s=_state_scale(X,state_scale)
    Z=X/s
    a=np.abs(Z); sg=np.sign(Z)
    b=np.where((a>c)&(a<=delta),sg,0.0)
    tau=np.where(a>delta,sg,0.0)
    K=kernel(p,chi)
    BB=b.T@b; TT=tau.T@tau; BT=b.T@tau
    G=K[0,0]*BB+K[1,1]*TT+K[0,1]*(BT+BT.T)
    C0=_normalise(G)
    ev0=np.linalg.eigvalsh(C0)
    rank0=int(np.linalg.matrix_rank(C0,tol=1e-8))
    C,alpha=condition_correlation(C0,target_min_eig)
    diag={
        'conditioning_alpha':alpha,
        'pre_lambda_min':float(ev0.min()),
        'rank_pre_conditioning':rank0,
        'gated_share':float(np.mean((b==0)&(tau==0))),
        'body_share':float(np.mean(b!=0)),
        'tail_share':float(np.mean(tau!=0)),
        'kernel_rank':int(np.linalg.matrix_rank(K,tol=1e-12)),
    }
    return (C,diag) if return_diagnostics else C


def covariance(X, *, c: float, delta: float, p: float, chi: float,
               state_scale: str='mad', target_min_eig: float=TARGET_MIN_EIG,
               return_diagnostics: bool=False):
    X=np.asarray(X,float)
    C,diag=correlation(X,c=c,delta=delta,p=p,chi=chi,state_scale=state_scale,
                       target_min_eig=target_min_eig,return_diagnostics=True)
    # Separate covariance lift: ordinary sample SD, irrespective of state scale.
    sd=np.clip(np.std(X,axis=0,ddof=1),1e-12,None)
    S=np.outer(sd,sd)*C
    return (S,diag) if return_diagnostics else S
