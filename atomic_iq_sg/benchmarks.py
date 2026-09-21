from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

def gerber_cov_stat(rets: np.array, threshold: float):
    n, p    = rets.shape
    sd_vec  = rets.std(axis=0)
    SD      = np.diag(sd_vec)
    cor_mat = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1):
            pos, neg, nn = 0, 0, 0
            for k in range(n):
                if ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])) or \
                   ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])):
                    pos += 1
                elif ((rets[k, i] >= threshold * sd_vec[i]) and (rets[k, j] <= -threshold * sd_vec[j])) or \
                     ((rets[k, i] <= -threshold * sd_vec[i]) and (rets[k, j] >= threshold * sd_vec[j])):
                    neg += 1
                elif abs(rets[k, i]) < threshold * sd_vec[i] and abs(rets[k, j]) < threshold * sd_vec[j]:
                    nn += 1
            cor_mat[i, j] = (pos - neg) / (n - nn)
            cor_mat[j, i] = cor_mat[i, j]
    cov_mat = SD @ cor_mat @ SD
    return cov_mat, cor_mat


def cov1Para(Y, k=None):
    N, p = Y.shape
    if k is None or math.isnan(k):
        Y = Y.sub(Y.mean(axis=0), axis=1); k = 1
    n = N - k
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy())) / n
    diag = np.diag(sample.to_numpy()); meanvar = sum(diag) / len(diag); target = meanvar * np.eye(p)
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(), Y.to_numpy()))
    sample2 = pd.DataFrame(np.matmul(Y2.T.to_numpy(), Y2.to_numpy())) / n
    piMat = pd.DataFrame(sample2.to_numpy() - np.multiply(sample.to_numpy(), sample.to_numpy()))
    pihat = sum(piMat.sum()); gammahat = np.linalg.norm(sample.to_numpy() - target, ord='fro') ** 2
    rhohat = 0; kappahat = (pihat-rhohat)/gammahat; shrinkage=max(0,min(1,kappahat/n))
    return shrinkage*target+(1-shrinkage)*sample


def cov2Para(Y, k=None):
    N,p=Y.shape
    if k is None or math.isnan(k): Y=Y.sub(Y.mean(axis=0),axis=1); k=1
    n=N-k; sample=pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n
    diag=np.diag(sample.to_numpy()); meanvar=sum(diag)/len(diag)
    meancov=(np.sum(sample.to_numpy())-np.sum(np.eye(p)*sample.to_numpy()))/(p*(p-1))
    target=pd.DataFrame(meanvar*np.eye(p)+meancov*(1-np.eye(p)))
    Y2=pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2=pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat=sum(piMat.sum()); gammahat=np.linalg.norm(sample.to_numpy()-target,ord='fro')**2
    rho_diag=(sample2.sum().sum()-np.trace(sample.to_numpy())**2)/p
    sum1,sum2=Y.sum(axis=1),Y2.sum(axis=1); temp=np.multiply(sum1.to_numpy(),sum1.to_numpy())-sum2
    rho_off1=np.sum(np.multiply(temp,temp))/(p*n)
    rho_off2=(sample.sum().sum()-np.trace(sample.to_numpy()))**2/p
    rho_off=(rho_off1-rho_off2)/(p-1); rhohat=rho_diag+rho_off
    kappahat=(pihat-rhohat)/gammahat; shrinkage=max(0,min(1,kappahat/n))
    return shrinkage*target+(1-shrinkage)*sample


def covCor(Y,k=None):
    N,p=Y.shape
    if k is None or math.isnan(k): Y=Y.sub(Y.mean(axis=0),axis=1); k=1
    n=N-k; sample=pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n
    samplevar=np.diag(sample.to_numpy()); sqrtvar=pd.DataFrame(np.sqrt(samplevar))
    rBar=(np.sum(np.sum(sample.to_numpy()/np.matmul(sqrtvar.to_numpy(),sqrtvar.T.to_numpy())))-p)/(p*(p-1))
    target=pd.DataFrame(rBar*np.matmul(sqrtvar.to_numpy(),sqrtvar.T.to_numpy())); target[np.eye(p).astype(bool)]=sample[np.eye(p).astype(bool)]
    Y2=pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy())); sample2=pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy())); pihat=sum(piMat.sum())
    gammahat=np.linalg.norm(sample.to_numpy()-target,ord='fro')**2; rho_diag=np.sum(np.diag(piMat))
    term1=pd.DataFrame(np.matmul((Y**3).T.to_numpy(),Y.to_numpy())/n)
    term2=pd.DataFrame(np.transpose(np.tile(samplevar,(p,1)))*sample); thetaMat=term1-term2; thetaMat[np.eye(p).astype(bool)]=0
    rho_off=rBar*(np.matmul((1/sqrtvar).to_numpy(),sqrtvar.T.to_numpy())*thetaMat).sum().sum()
    rhohat=rho_diag+rho_off; kappahat=(pihat-rhohat)/gammahat; shrinkage=max(0,min(1,kappahat/n))
    return shrinkage*target+(1-shrinkage)*sample


def covDiag(Y,k=None):
    N,p=Y.shape
    if k is None or math.isnan(k): Y=Y.sub(Y.mean(axis=0),axis=1); k=1
    n=N-k; sample=pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n
    target=pd.DataFrame(np.diag(np.diag(sample.to_numpy()))); Y2=pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2=pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n; piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat=sum(piMat.sum()); gammahat=np.linalg.norm(sample.to_numpy()-target,ord='fro')**2; rho_diag=np.sum(np.diag(piMat))
    kappahat=(pihat-rho_diag)/gammahat; shrinkage=max(0,min(1,kappahat/n))
    return shrinkage*target+(1-shrinkage)*sample


def covMarket(Y,k=None):
    N,P=Y.shape
    if k is None or math.isnan(k): Y=Y.sub(Y.mean(axis=0),axis=1); k=1
    n=N-k; sample=pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n
    Ymkt=Y.mean(axis=1); covmkt=pd.DataFrame(np.matmul(Y.T.to_numpy(),Ymkt.to_numpy()))/n
    varmkt=np.matmul(Ymkt.T.to_numpy(),Ymkt.to_numpy())/n
    target=pd.DataFrame(np.matmul(covmkt.to_numpy(),covmkt.T.to_numpy()))/varmkt; target[np.eye(P).astype(bool)]=sample[np.eye(P).astype(bool)]
    Y2=pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy())); sample2=pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy())); pihat=sum(piMat.sum())
    gammahat=np.linalg.norm(sample.to_numpy()-target,ord='fro')**2; rho_diag=np.sum(np.diag(piMat))
    temp=Y*pd.DataFrame([Ymkt for _ in range(P)]).T; temp=temp.iloc[:,P:]; covmktSQ=pd.DataFrame([covmkt[0] for _ in range(P)])
    v1=pd.DataFrame((1/n)*np.matmul(Y2.T.to_numpy(),temp.to_numpy())-np.multiply(covmktSQ.T.to_numpy(),sample.to_numpy()))
    roff1=(np.sum(np.multiply(v1.to_numpy(),covmktSQ.to_numpy()))-np.sum(np.diag(np.multiply(v1.to_numpy(),covmkt.to_numpy()))))/varmkt
    v3=pd.DataFrame((1/n)*np.matmul(temp.T.to_numpy(),temp.to_numpy())-varmkt*sample)
    roff3=(np.sum(np.multiply(v3.to_numpy(),np.matmul(covmkt.to_numpy(),covmkt.T.to_numpy())))-np.sum(np.multiply(np.diag(v3.to_numpy()),covmkt[0]**2)))/varmkt**2
    rhohat=rho_diag+2*roff1-roff3; kappahat=(pihat-rhohat)/gammahat; shrinkage=max(0,min(1,kappahat/n))
    return shrinkage*target+(1-shrinkage)*sample


def GIS(Y,k=None):
    N,p=Y.shape
    if k is None or math.isnan(k): Y=Y.sub(Y.mean(axis=0),axis=1); k=1
    n=N-k; c=p/n; sample=pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n; sample=(sample+sample.T)/2
    lambda1,u=np.linalg.eigh(sample); lambda1=lambda1.real.clip(min=0); dfu=pd.DataFrame(u,columns=lambda1); dfu.sort_index(axis=1,inplace=True); lambda1=dfu.columns
    h=(min(c**2,1/c**2)**0.35)/p**0.35; invlambda=1/lambda1[max(1,p-n+1)-1:p]
    dfl=pd.DataFrame({'lambda':invlambda}); Lj=pd.DataFrame(dfl[np.repeat(dfl.columns.values,min(p,n))].to_numpy()); Lj_i=Lj.subtract(Lj.T)
    theta=Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj)*h**2)).mean(axis=0)
    Htheta=Lj.multiply(Lj*h).div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj)*h**2)).mean(axis=0); Atheta2=theta**2+Htheta**2
    if p<=n:
        deltahat_1=(1-c)*invlambda+2*c*invlambda*theta
        delta=1/((1-c)**2*invlambda+2*c*(1-c)*invlambda*theta+c**2*invlambda*Atheta2); delta=delta.to_numpy()
    else: return -1
    temp=pd.DataFrame(deltahat_1); x=min(invlambda); temp.loc[temp[0]<x,0]=x; deltaLIS_1=temp[0]
    return dfu.to_numpy()@np.diag((delta/deltaLIS_1)**0.5)@dfu.T.to_numpy().conjugate()


def LIS(Y,k=None):
    N,p=Y.shape
    if k is None or math.isnan(k): Y=Y.sub(Y.mean(axis=0),axis=1); k=1
    n=N-k; c=p/n; sample=pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n; sample=(sample+sample.T)/2
    lambda1,u=np.linalg.eigh(sample); lambda1=lambda1.real.clip(min=0); dfu=pd.DataFrame(u,columns=lambda1); dfu.sort_index(axis=1,inplace=True); lambda1=dfu.columns
    h=(min(c**2,1/c**2)**0.35)/p**0.35; invlambda=1/lambda1[max(1,p-n+1)-1:p]
    dfl=pd.DataFrame({'lambda':invlambda}); Lj=pd.DataFrame(dfl[np.repeat(dfl.columns.values,min(p,n))].to_numpy()); Lj_i=Lj.subtract(Lj.T)
    theta=Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj)*h**2)).mean(axis=0)
    if p<=n: deltahat_1=(1-c)*invlambda+2*c*invlambda*theta
    else: return -1
    temp=pd.DataFrame(deltahat_1); x=min(invlambda); temp.loc[temp[0]<x,0]=x; deltaLIS_1=temp[0]
    return dfu.to_numpy()@np.diag(1/deltaLIS_1)@dfu.T.to_numpy().conjugate()


def QIS(Y,k=None):
    N,p=Y.shape
    if k is None or math.isnan(k): Y=Y.sub(Y.mean(axis=0),axis=1); k=1
    n=N-k; c=p/n; sample=pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n; sample=(sample+sample.T)/2
    lambda1,u=np.linalg.eigh(sample); lambda1=lambda1.real.clip(min=0); dfu=pd.DataFrame(u,columns=lambda1); dfu.sort_index(axis=1,inplace=True); lambda1=dfu.columns
    h=(min(c**2,1/c**2)**0.35)/p**0.35; invlambda=1/lambda1[max(1,p-n+1)-1:p]
    dfl=pd.DataFrame({'lambda':invlambda}); Lj=pd.DataFrame(dfl[np.repeat(dfl.columns.values,min(p,n))].to_numpy()); Lj_i=Lj.subtract(Lj.T)
    theta=Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj)*h**2)).mean(axis=0)
    Htheta=Lj.multiply(Lj*h).div(Lj_i.multiply(Lj_i).add(Lj.multiply(Lj)*h**2)).mean(axis=0); Atheta2=theta**2+Htheta**2
    if p<=n:
        delta=1/((1-c)**2*invlambda+2*c*(1-c)*invlambda*theta+c**2*invlambda*Atheta2); delta=delta.to_numpy()
    else:
        delta0=1/((c-1)*np.mean(invlambda.to_numpy())); delta=np.concatenate((np.repeat(delta0,p-n),1/(invlambda*Atheta2)),axis=None)
    deltaQIS=delta*(sum(lambda1)/sum(delta))
    return dfu.to_numpy()@np.diag(deltaQIS)@dfu.T.to_numpy().conjugate()


def mpPDF(var,q,pts):
    eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    eVal=np.linspace(eMin,eMax,pts); return pd.Series(q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5,index=eVal)


def getPCA(matrix):
    eVal,eVec=np.linalg.eig(matrix); indices=eVal.argsort()[::-1]; eVal,eVec=eVal[indices],eVec[:,indices]; return np.diagflat(eVal),eVec


def fitKDE(obs,bWidth=.15,kernel='gaussian',x=None):
    if len(obs.shape)==1: obs=obs.reshape(-1,1)
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
    if x is None: x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1: x=x.reshape(-1,1)
    return pd.Series(np.exp(kde.score_samples(x)),index=x.flatten())


def cov2corr(cov):
    std=np.sqrt(np.diag(cov)); corr=cov/np.outer(std,std); corr[corr<-1],corr[corr>1]=-1,1; return corr


def corr2cov(corr,std): return corr*np.outer(std,std)


def errPDFs(var,eVal,q,bWidth,pts=1000):
    var=var[0]; pdf0=mpPDF(var,q,pts); pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values); return np.sum((pdf1-pdf0)**2)


def findMaxEval(eVal,q,bWidth):
    out=minimize(lambda *x:errPDFs(*x),x0=np.array(0.5),args=(eVal,q,bWidth),bounds=((1E-5,1-1E-5),))
    var=out['x'][0] if out['success'] else 1; return var*(1+(1./q)**.5)**2,var


def denoisedCorr(eVal,eVec,nFacts):
    eVal_=np.diag(eVal).copy(); eVal_[nFacts:]=eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
    return cov2corr(eVec@np.diag(eVal_)@eVec.T)


def denoisedCorr2(eVal,eVec,nFacts,alpha):
    eValL,eVecL=eVal[:nFacts,:nFacts],eVec[:,:nFacts]; eValR,eVecR=eVal[nFacts:,nFacts:],eVec[:,nFacts:]
    corr0=eVecL@eValL@eVecL.T; corr1=eVecR@eValR@eVecR.T
    return corr0+alpha*corr1+(1-alpha)*np.diag(np.diag(corr1))


def RMT1(rets:np.array,q=None,bWidth=0.01):
    rets=np.asarray(rets,float); q=rets.shape[0]/rets.shape[1] if q is None else q
    cov0=np.cov(rets,rowvar=0); corr0=cov2corr(cov0); eVal0,eVec0=getPCA(corr0); eMax0,_=findMaxEval(np.diag(eVal0),q,bWidth)
    nFacts0=eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0); cor_mat=denoisedCorr(eVal0,eVec0,nFacts0)
    return corr2cov(cor_mat,np.diag(cov0)**.5),cor_mat


def RMT2(rets:np.array,q=None,bWidth=0.01,alpha=0.1):
    rets=np.asarray(rets,float); q=rets.shape[0]/rets.shape[1] if q is None else q
    cov0=np.cov(rets,rowvar=0); corr0=cov2corr(cov0); eVal0,eVec0=getPCA(corr0); eMax0,_=findMaxEval(np.diag(eVal0),q,bWidth)
    nFacts0=eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0); cor_mat=denoisedCorr2(eVal0,eVec0,nFacts0,alpha)
    return corr2cov(cor_mat,np.diag(cov0)**.5),cor_mat
