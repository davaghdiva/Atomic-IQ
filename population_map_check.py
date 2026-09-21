"""Reproduce the Atomic-IQ population-map illustration and optional accuracy check.

This script does not use the empirical return panel. It evaluates the Atomic-IQ
population functional at the frozen parameter point under standardized bivariate
Gaussian and t(5) models using scrambled Sobol quasi-Monte Carlo draws. The sign
boundary is analytic: 2/pi * arcsin(rho).
"""
import argparse, math
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc, chi2, t

C, DELTA, P, CHI = 0.40, 1.25, 0.10, 0.90
CROSS = CHI * math.sqrt(P * (1.0 - P))
NORMAL_MAD_CONSISTENCY = 1.482602218505602
RHO_TABLE = [0.10, 0.30, 0.50, 0.70, 0.90]

def aiq_from_samples(x,y):
    ax,ay=np.abs(x),np.abs(y); bx=np.sign(x)*((ax>C)&(ax<=DELTA)); by=np.sign(y)*((ay>C)&(ay<=DELTA))
    tx=np.sign(x)*(ax>DELTA); ty=np.sign(y)*(ay>DELTA)
    hij=np.mean(P*bx*by+(1-P)*tx*ty+CROSS*(bx*ty+tx*by))
    hii=np.mean(P*(bx!=0)+(1-P)*(tx!=0)); hjj=np.mean(P*(by!=0)+(1-P)*(ty!=0))
    return float(hij/math.sqrt(hii*hjj))

def fixed_draws(seed_g=20260922,seed_t=20260921,power=19):
    ug=np.clip(qmc.Sobol(2,scramble=True,seed=seed_g).random_base2(power),1e-12,1-1e-12); g1,g2=ndtri(ug[:,0]),ndtri(ug[:,1])
    ut=np.clip(qmc.Sobol(3,scramble=True,seed=seed_t).random_base2(power),1e-12,1-1e-12); z1,z2=ndtri(ut[:,0]),ndtri(ut[:,1])
    w=np.sqrt(chi2.ppf(ut[:,2],5)/5.0); scale=NORMAL_MAD_CONSISTENCY*t.ppf(0.75,5); return g1,g2,z1,z2,w,scale,z1/w/scale

def table_values(seed_g=20260922,seed_t=20260921,power=19):
    g1,g2,z1,z2,w,s,xt=fixed_draws(seed_g,seed_t,power); out=[]
    for rho in RHO_TABLE:
        gy=rho*g1+math.sqrt(1-rho*rho)*g2; ty=(rho*z1+math.sqrt(1-rho*rho)*z2)/w/s
        out.append((rho,aiq_from_samples(g1,gy),aiq_from_samples(xt,ty),2/math.pi*math.asin(rho)))
    return out

def accuracy_check(scrambles=20,power=19):
    vg={r:[] for r in RHO_TABLE}; vt={r:[] for r in RHO_TABLE}
    for k in range(scrambles):
        for rho,g,tv,_ in table_values(31000+k,41000+k,power): vg[rho].append(g); vt[rho].append(tv)
    print(f'accuracy check: {scrambles} independent scrambles, 2^{power} points each')
    for rho in RHO_TABLE: print(f'rho={rho:.2f}: Gaussian sd={np.std(vg[rho],ddof=1):.8f}, t5 sd={np.std(vt[rho],ddof=1):.8f}')

def main(do_accuracy=False):
    print('rho,AIQ_Gaussian,AIQ_t5,sign_boundary')
    for row in table_values(): print(f'{row[0]:.2f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}')
    g1,g2,z1,z2,w,s,xt=fixed_draws(); grid=np.linspace(-.99,.99,199); ag=[]; at=[]
    for rho in grid:
        ag.append(aiq_from_samples(g1,rho*g1+math.sqrt(1-rho*rho)*g2))
        at.append(aiq_from_samples(xt,(rho*z1+math.sqrt(1-rho*rho)*z2)/w/s))
    print('Gaussian monotone on grid:',bool(np.all(np.diff(ag)>0))); print('t(5) monotone on grid:',bool(np.all(np.diff(at)>0)))
    print('max |Gaussian-t5| on grid:',float(np.max(np.abs(np.asarray(ag)-np.asarray(at)))))
    if do_accuracy: accuracy_check()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--accuracy',action='store_true'); a=p.parse_args(); main(a.accuracy)
