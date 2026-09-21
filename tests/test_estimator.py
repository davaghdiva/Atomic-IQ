import numpy as np
from atomic_iq_sg.estimator import kernel,correlation,covariance

def test_kernel_psd_box():
    for p in np.linspace(0,1,11):
        for chi in np.linspace(0,1,11): assert np.linalg.eigvalsh(kernel(p,chi)).min()>=-1e-12

def test_identical_and_opposite():
    x=np.linspace(-2,2,20); X=np.column_stack([x,x,-x]); C=correlation(X,c=.4,delta=1.25,p=.1,chi=.9,state_scale='mad',target_min_eig=0.0); assert abs(C[0,1]-1)<1e-12; assert abs(C[0,2]+1)<1e-12

def test_universe_invariance():
    rng=np.random.default_rng(1); X=rng.normal(size=(20,2)); C1=correlation(X,c=.4,delta=1.25,p=.1,chi=.9,state_scale='mad',target_min_eig=0.0); X2=np.column_stack([X,rng.normal(0,50,size=(20,3))]); C2=correlation(X2,c=.4,delta=1.25,p=.1,chi=.9,state_scale='mad'); assert np.allclose(C1,C2[:2,:2])

def test_covariance_lift_uses_sd():
    rng=np.random.default_rng(2); X=rng.normal(size=(20,4)); S=covariance(X,c=.4,delta=1.25,p=.1,chi=.9,state_scale='mad'); assert np.allclose(np.diag(S),np.var(X,axis=0,ddof=1))
