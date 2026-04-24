"""
Name    : aiq_cov.py
Author  : William Smyth & Layla Abu Khalaf
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : atomic-IQ covariance builders (AIQ1 & AIQ2).
"""
import numpy as np
import pandas as pd
from numba import njit, prange

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _softmax(z):
    z = np.asarray(z, dtype=float).ravel()
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def _weights_from_logits(logits, mode="free"):
    """
    returns (wB, wW, wT, wI).
    - "free"   : three independent channel weights via sigmoid in (0,1); wI=1-sum.
    - "simplex": softmax over 4 (or 3) logits (legacy). If only 3 are given, wI=0.
    - "clipped": treat first 3 entries as already in R, clip to [0,1]; wI residual.
    """
    z = np.asarray(logits, dtype=float).ravel()
    if mode == "free":
        if z.size < 3:
            raise ValueError("weight_mode='free' expects at least 3 logits.")
        wB, wW, wT = _sigmoid(z[:3])
        wI = 1.0 - (wB + wW + wT)
    elif mode == "simplex":
        p = _softmax(z)
        if p.size >= 4:
            wB, wW, wT, wI = p[:4]
        elif p.size == 3:
            wB, wW, wT = p
            wI = 0.0
        else:
            raise ValueError("Need 3 or 4 logits for weight_mode='simplex'.")
    elif mode == "clipped":
        if z.size < 3:
            raise ValueError("weight_mode='clipped' expects at least 3 values.")
        wB, wW, wT = np.clip(z[:3], 0.0, 1.0)
        wI = 1.0 - (wB + wW + wT)
    elif mode == "basic_eta":
        if z.size < 1:
            raise ValueError("weight_mode='basic_eta' expects 1 logit.")
        eta = _sigmoid(z[0])
        wB  = eta
        wW  = eta * eta
        wT  = 1.0
        wI  = 1.0 - (wB + wW + wT)
    else:
        raise ValueError(f"Unknown weight_mode: {mode}")
    return float(wB), float(wW), float(wT), float(wI)

def _temporal_factors_np(T, gamma, epsil, mode="signed", gamma_max=None, eps_floor=0.0, schur_safe=False):
    """time-decay factors a_t in (0,1], optionally normalised for Schur product safety (AIQ2)."""
    gamma  = float(gamma)
    epsil  = float(epsil)
    if gamma_max is not None and gamma_max > 0:
        gamma = float(np.clip(gamma, -gamma_max, gamma_max))
    g_abs = abs(gamma)

    t = np.arange(T, dtype=float)
    if mode == "raw":
        age = (T - 1 - t)
        age = np.maximum(age - epsil, 0.0)
        a = np.exp(-gamma * age)  # may exceed 1 if gamma < 0
    elif mode == "signed":
        is_recent  = (gamma >= 0.0)
        age_choice = (T - 1 - t) if is_recent else t
        age = np.maximum(age_choice - epsil, 0.0)
        a = np.exp(-g_abs * age) # in (0,1]
    else:
        raise ValueError("gamma mode must be 'raw' or 'signed'")

    if schur_safe:
        m = np.max(a)
        if m > 0:
            a = a / m  # ensure max(a)==1 for Schur product safety

    if eps_floor and eps_floor > 0:
        a = eps_floor + (1.0 - eps_floor) * a
    return a

def _to_corr_np(G, eps=1e-12):
    """
    convert accumulator G to correlation 
    matrix C with safe diagonal handling.
    """
    G = 0.5 * (G + G.T)
    d = np.diag(G).copy()
    mask = d <= eps
    d_safe = d.copy()
    d_safe[mask] = 1.0
    inv_sqrt = 1.0 / np.sqrt(d_safe)
    C = (G * inv_sqrt).T * inv_sqrt 
    if np.any(mask):
        idx = np.where(mask)[0]
        C[idx, :]   = 0.0
        C[:, idx]   = 0.0
        C[idx, idx] = 1.0
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    C = np.clip(C, -1.0, 1.0)
    return C

def _prep_X(df_returns: pd.DataFrame, center_method="mean"):
    X = df_returns.values.astype(float)  # T x N
    if center_method == "mean":
        mean_vec = np.mean(X, axis=0, keepdims=True)
    elif center_method == "median":
        mean_vec = np.median(X, axis=0, keepdims=True)
    elif center_method == "zero":
        mean_vec = np.zeros((1, X.shape[1]), dtype=float)
    else:
        raise ValueError("center_method must be 'mean','median','zero'")
    R = X - mean_vec
    return R

def _stds_from_R(R, eps=1e-12):
    stds = np.std(R, axis=0)
    return np.clip(stds, eps, None)

def _pair_scale_scalar(si: float, sj: float, mode: str, eps: float = 1e-12) -> float:
    si = float(max(si, eps)); sj = float(max(sj, eps))
    if mode == "pair_min":
        return min(si, sj)
    elif mode == "pair_avg":
        return 0.5 * (si + sj)
    elif mode == "pair_max":
        return max(si, sj)
    else:
        raise ValueError(f"Unknown pairwise scale_method: {mode}")

def atomic_covariance(
    df_returns: pd.DataFrame,
    logits, delta, gamma, epsilon,
    iq_method="iq1", center_method="mean", scale_method="vols", threshold=0.05,
    weight_mode="free",     # "free","simplex","clipped"
    psd_repair="auto",      # "auto" (tight),"model_free","off"
    target_min_eig=None,    # lambda_min >= lambda*, in correlation scale
    target_kappa=None,      # condition number <= kappa*
    tol_psd=1e-08,          # numerical tolerance
    PSD_check=True, gamma_mode="signed", gamma_max=0.10, a_floor=0.0
) -> pd.DataFrame:
    """
    build atomic-IQ covariance allowing channel weights in [0,1] independently (free sum).
    If xi = wB + wW + wT > 1, apply the tight analytic eigenfloor when needed so that
    the final correlation S is PSD while remaining in the squeezing representation.

    ref: canonical form S = (1-xi)I + xiP, exact PSD test for xi>1, and tight eigenfloor t: 
    """

    R    = _prep_X(df_returns, center_method=center_method)
    T, N = R.shape
    vols = _stds_from_R(R)
    thr  = float(threshold)
    delt = float(delta)

    a = _temporal_factors_np(
        T, float(gamma), float(epsilon),
        mode=gamma_mode, gamma_max=gamma_max, eps_floor=a_floor,
        schur_safe=(iq_method == "iq2")
    )

    G_B = np.zeros((N, N), dtype=float)
    G_W = np.zeros((N, N), dtype=float)
    G_T = np.zeros((N, N), dtype=float)

    for i in range(N):
        for j in range(i, N):
            if scale_method in ("pair_min", "pair_avg", "pair_max"):
                s_ij = _pair_scale_scalar(vols[i], vols[j], scale_method)
                xi = R[:, i] / s_ij
                xj = R[:, j] / s_ij
            elif scale_method in ("vols", "vols_max", "vols_avg", "vols_min"):
                
                if scale_method == "vols":
                    di, dj = vols[i], vols[j]
                else:
                    g = {"vols_max": float(vols.max()),
                         "vols_avg": float(vols.mean()),
                         "vols_min": float(max(vols.min(), 1e-12))}[scale_method]
                    di = dj = g
                xi = R[:, i] / di
                xj = R[:, j] / dj
            else:
                raise ValueError("scale_method must be one of "
                                 "{'pair_min','pair_avg','pair_max','vols','vols_max','vols_avg','vols_min'}")
    
            axi = np.abs(xi);  axj = np.abs(xj)
            sgn = np.sign(xi) * np.sign(xj)
    
            black = (axi <= thr) | (axj <= thr)
            body  = (~black) & (axi <= delt) & (axj <= delt)
            wing  = (~black) & ((axi <= delt) ^ (axj <= delt))
            tail  = (~black) & (axi >  delt) & (axj >  delt)

            if iq_method == "iq1":
                if np.any(body):
                    w = a[body]; sb = sgn[body]; ssum = float(np.sum(w))
                    G_B[i, i] += ssum;  G_B[j, j] += (ssum if j != i else 0.0)
                    if j != i:
                        G_B[i, j] += float(np.sum(w * sb)); G_B[j, i] = G_B[i, j]
                if np.any(wing):
                    w = a[wing]; sw = sgn[wing]; ssum = float(np.sum(w))
                    G_W[i, i] += ssum;  G_W[j, j] += (ssum if j != i else 0.0)
                    if j != i:
                        G_W[i, j] += float(np.sum(w * sw)); G_W[j, i] = G_W[i, j]
                if np.any(tail):
                    w = a[tail]; st = sgn[tail]; ssum = float(np.sum(w))
                    G_T[i, i] += ssum;  G_T[j, j] += (ssum if j != i else 0.0)
                    if j != i:
                        G_T[i, j] += float(np.sum(w * st)); G_T[j, i] = G_T[i, j]
            else:  # iq2
                if np.any(body):
                    cnt = float(np.sum(body))
                    G_B[i, i] += cnt;  G_B[j, j] += (cnt if j != i else 0.0)
                    if j != i:
                        G_B[i, j] += float(np.sum(a[body] * sgn[body])); G_B[j, i] = G_B[i, j]
                if np.any(wing):
                    cnt = float(np.sum(wing))
                    G_W[i, i] += cnt;  G_W[j, j] += (cnt if j != i else 0.0)
                    if j != i:
                        G_W[i, j] += float(np.sum(a[wing] * sgn[wing])); G_W[j, i] = G_W[i, j]
                if np.any(tail):
                    cnt = float(np.sum(tail))
                    G_T[i, i] += cnt;  G_T[j, j] += (cnt if j != i else 0.0)
                    if j != i:
                        G_T[i, j] += float(np.sum(a[tail] * sgn[tail])); G_T[j, i] = G_T[i, j]

    C_B, C_W, C_T = _to_corr_np(G_B), _to_corr_np(G_W), _to_corr_np(G_T)
    # each C_α is correlation-PSD by atomic construction + one normalisation.

    wB, wW, wT, wI = _weights_from_logits(logits, mode=weight_mode)
    xi = wB + wW + wT  # collective squeeze

    # structured component and raw S
    I  = np.eye(N, dtype=float)
    S_struct = wB * C_B + wW * C_W + wT * C_T
    S = S_struct + wI * I  # equals (1 - xi) I + xi P when xi > 0 

    # exact feasibility + tight eigenfloor 
    t_used = 0.0
    if xi > 1.0 and psd_repair != "off":
        # exact PSD test with mu = lambda_min(P)
        # if xi == 0, S = I; if xi > 0, form P safely
        P = S_struct / xi
        mu = float(np.min(np.linalg.eigvalsh(0.5 * (P + P.T))))  # lambda_min(P)
        feasible = (1.0 - xi) + xi * mu >= -1e-12  # small tolerance

        if not feasible:
            if psd_repair == "model_free":
                # conservative floor using only xi
                t_used = max(0.0, 1.0 - 1.0 / xi)
            else:
                # tight analytic floor using mu
                # t >= max{0, 1-1/[xi(1 - mu)]}
                denom = xi * (1.0 - mu)
                if denom <= 0:
                    t_used = 1.0  # pathological (mu >= 1). clamp to identity.
                else:
                    t_used = max(0.0, 1.0 - 1.0 / denom)
            # apply eigenfloor: S-tilde = (1-t)S + tI
            S = (1.0 - t_used) * S + t_used * I
            xi_eff = (1.0 - t_used) * xi  # new collective squeeze inside squeezing family

    # optional conditioning goals (use after PSD secured)
    if target_min_eig is not None or target_kappa is not None:
        evals = np.linalg.eigvalsh(0.5 * (S + S.T))
        alpha, beta = float(np.max(evals)), float(np.min(evals))
        # bump lambda_min to target_min_eig if requested
        t1 = 0.0
        if target_min_eig is not None and beta < target_min_eig:
            # (1-t)beta + t >= lambda*  =>  t >= (lambda*-beta)/(1-beta)
            t1 = max(0.0, min(1.0, (target_min_eig - beta) / (1.0 - beta + 1e-15)))
        # shrink condition number to target_kappa if requested
        t2 = 0.0
        if target_kappa is not None and alpha > 0 and beta > 0:
            # ((1-t)alpha + t)/((1-t)beta + t) <= kappa*
            num = alpha - target_kappa * beta
            den = (alpha - 1.0) + target_kappa * (1.0 - beta)
            if den > 0:
                t2 = max(0.0, min(1.0, num / den))
        t_extra = max(t1, t2)
        if t_extra > 0:
            S = (1.0 - t_extra) * S + t_extra * I
            t_used = t_used + (1.0 - t_used) * t_extra  # cumulative floor

    if PSD_check:
        ev = np.linalg.eigvalsh(0.5 * (S + S.T))
        if np.min(ev) < -1e-08:
            print(f"[aiq_cov] WARNING: C_IQ not PSD (min eig={np.min(ev):.3e}); proceeding.")

    SD = np.diag(vols)
    Sigma = SD @ S @ SD
    return pd.DataFrame(Sigma, index=df_returns.columns, columns=df_returns.columns)
