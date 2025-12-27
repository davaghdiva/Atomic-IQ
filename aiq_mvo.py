"""
Name    : aiq_mvo.py
Author  : William Smyth & Layla Abu Khalaf
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : atomic-IQ parameter search + Max-Sharpe backtest.
"""
import json
import os
import numpy as np
import pandas as pd
from aiq_cov import atomic_covariance
from dataclasses import dataclass
from defaults import DEFAULTS

try:
    import optuna
    _HAVE_OPTUNA = True
except Exception:
    optuna = None
    _HAVE_OPTUNA = False

def _script_dir() -> str:
    return os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()

def _ensure_monthly_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df_prices.index, pd.DatetimeIndex):
        raise ValueError("Price CSV must have a 'Date' index column parsable to dates.")
    return df_prices.resample("M", label="right").last()

def _load_prices_subset(csv_path: str, n_assets: int | None = None) -> pd.DataFrame:
    path = csv_path if os.path.isabs(csv_path) else os.path.join(_script_dir(), csv_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find price file at {path}.")
    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=False).set_index("Date")
    if n_assets is not None and n_assets > 0:
        df = df.iloc[:, :n_assets].copy()
    return _ensure_monthly_prices(df)

def _load_csv_series(csv_path: str, column: str | None = None) -> pd.Series:
    path = csv_path if os.path.isabs(csv_path) else os.path.join(_script_dir(), csv_path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    ser = df.iloc[:, 0] if column is None else df[column]
    ser = pd.to_numeric(ser, errors="coerce").dropna()
    ser.index = pd.to_datetime(ser.index)
    return ser.sort_index()

def _asof_align(series: pd.Series, idx: pd.DatetimeIndex, method="ffill") -> pd.Series:
    union = series.index.union(idx).sort_values()
    s = series.reindex(union)
    if method == "ffill":
        s = s.ffill()
    elif method == "bfill":
        s = s.bfill()
    return s.reindex(idx)

def _load_rf_aligned(idx: pd.DatetimeIndex) -> pd.Series | None:
    try:
        rf = _load_csv_series(DEFAULTS['rf_csv'], column=DEFAULTS['rf_column'])
        if DEFAULTS['rf_interpretation'] == "annualized_yield":
            y = rf / 100.0
            ppy = 12 if str(DEFAULTS['rf_frequency']).upper().startswith("M") else 252
            rf = (1.0 + y) ** (1.0 / ppy) - 1.0
        rf_aligned = _asof_align(rf, idx, method=DEFAULTS['rf_align_method'])
        return rf_aligned.rename("rf")
    except Exception:
        return None

def annualized_sharpe(ret: pd.Series, rf: pd.Series | float | None = None) -> float:
    if rf is None:
        ex = ret
    elif isinstance(rf, (int, float)):
        ex = ret - float(rf)
    else:
        ex = (ret - rf.reindex(ret.index)).dropna()
    mu = ex.mean()
    sd = ex.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((mu / sd) * np.sqrt(12.0))

@dataclass
class MVOSettings:
    min_w: float    =  0.0
    max_w: float    =  1.0
    cost_bps: float = 10.0 

def _max_sharpe_objective(weights: np.ndarray, mean_excess: np.ndarray, cov_monthly: np.ndarray) -> float:
    mu_ann = 12.0 * float(np.dot(mean_excess, weights))
    sd_ann = float(np.sqrt(np.dot(weights.T, np.dot(cov_monthly * 12.0, weights))))
    if sd_ann == 0 or np.isnan(sd_ann):
        return 1e6
    return - mu_ann / sd_ann

def solve_max_sharpe(
    returns_excess_window: pd.DataFrame,
    cov_monthly: np.ndarray,
    prev_weights: np.ndarray | None,
    settings: MVOSettings,
) -> np.ndarray:
    from scipy.optimize import minimize
    p = returns_excess_window.shape[1]
    w0 = np.full(p, 1.0 / p, dtype=float)
    bounds = tuple((settings.min_w, settings.max_w) for _ in range(p))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    mean_excess = returns_excess_window.mean().values.astype(float)

    def objective(w: np.ndarray) -> float:
        base = _max_sharpe_objective(w, mean_excess, cov_monthly)
        if prev_weights is not None:
            l1_turnover = np.abs(w - prev_weights).sum()
            base += (settings.cost_bps / 10000.0) * l1_turnover
        return base

    res = minimize(objective, x0=w0, method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x
    w[np.abs(w) < 1e-6] = 0.0
    s = w.sum()
    if s <= 0:
        w = np.full(p, 1.0 / p, dtype=float)
    else:
        w = w / s
    return w

def _bisection_method(f, a, b, tol=1e-5, max_iter=100):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("The function must have different signs at a and b.")
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        if fc * fa < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    raise ValueError("Maximum iterations reached without convergence.")

class TransCost:
    """self-financing transaction cost model (bps of traded notional)."""
    def __init__(self, c_bps: float):
        self.c = float(c_bps) / 10000.0  # transaction cost in bps
        self.tickers = None

    def _init_cost_ratio(self, new_w: list[float], old_w: list[float]) -> float:
        e = [(-1 if n_w > o_w else +1) for o_w, n_w in zip(old_w, new_w)]
        return (
            (1 - self.c * sum(w * ei for w, ei in zip(old_w, e))) /
            (1 - self.c * sum(w * ei for w, ei in zip(new_w, e)))
        )

    def _cost_func(self, new_w, old_w, cost):
        e = [(-1 if n_w > o_w else +1) for o_w, n_w in zip(old_w, new_w)]
        return 1 - self.c * sum((old_w[i] - new_w[i] * cost) * e[i] for i in range(len(new_w))) - cost

    def get_cost(self, new_weights: dict, old_weights: dict) -> float:
        # align tickers and normalise both weight vectors (if non-zero)
        self.tickers   = sorted(set(new_weights.keys()).union(old_weights.keys()))
        np_new_weights = [new_weights.get(t, 0) for t in self.tickers]
        np_old_weights = [old_weights.get(t, 0) for t in self.tickers]
    
        if sum(np_new_weights) != 0:
            s = float(sum(np_new_weights))
            np_new_weights = [w / s for w in np_new_weights]
        if sum(np_old_weights) != 0:
            s = float(sum(np_old_weights))
            np_old_weights = [w / s for w in np_old_weights]
    
        init_cost = self._init_cost_ratio(np_new_weights, np_old_weights)
        f = lambda c: self._cost_func(np_new_weights, np_old_weights, c)
    
        check = f(init_cost)
        if abs(check) < 1e-10:
            return 1 - init_cost
    
        # ensure root-finder has bracket. If no sign change on [0,1], fall
        # back to the endpoint with the smaller residual to avoid raising.
        f0 = f(0.0)
        f1 = f(1.0)
        if f0 * f1 >= 0:
            c_star = 0.0 if abs(f0) <= abs(f1) else 1.0
            return 1 - c_star
    
        root = _bisection_method(f, 0.0, 1.0)
        return 1 - root

@dataclass
class BacktestConfig:
    lookback: int
    cost_bps: float = 10.0
    initial_value: float = 100000.0

def backtest_aiq_mvo(
    prcs: pd.DataFrame,
    cov_params: dict,
    bt_cfg: BacktestConfig,
):
    """run monthly rebalanced Max Sharpe portfolio using atomic-IQ."""
    prcs_m = _ensure_monthly_prices(prcs.copy())
    rets_m = prcs_m.pct_change().dropna()
    dates  = rets_m.index
    symbols = prcs_m.columns.tolist()

    rf_m = _load_rf_aligned(dates)
    if rf_m is not None:
        rets_excess = rets_m.sub(rf_m, axis=0).dropna()
    else:
        rets_excess = rets_m.copy()

    tc = TransCost(c_bps=bt_cfg.cost_bps)
    values = []
    prev_weights = np.zeros(len(symbols), dtype=float)
    V = float(bt_cfg.initial_value)

    for t in range(bt_cfg.lookback, len(rets_m)):
        window_rets   = rets_m.iloc[t - bt_cfg.lookback: t]
        window_excess = rets_excess.iloc[t - bt_cfg.lookback: t]
        r_next = rets_m.iloc[t].values.astype(float)

        Sigma_df = atomic_covariance(
            window_rets,
            logits=cov_params["logits"],
            delta=cov_params["delta"],
            gamma=cov_params["gamma"],
            epsilon=cov_params["epsilon"],
            iq_method=cov_params["iq_method"],
            center_method=cov_params["center_method"],
            scale_method=cov_params["scale_method"],
            threshold=cov_params["threshold"],
            PSD_check=cov_params["PSD_check"],
            gamma_mode=cov_params["gamma_mode"],
            gamma_max=cov_params["u_gamma"],
            a_floor=cov_params["a_floor"],
            weight_mode=DEFAULTS["weight_mode"],
            psd_repair=DEFAULTS["psd_repair"],
            target_min_eig=DEFAULTS["target_min_eig"],
            target_kappa=DEFAULTS["target_kappa"],
        )
        cov_m = Sigma_df.values.astype(float)

        settings = MVOSettings(cost_bps=bt_cfg.cost_bps)
        w_new = solve_max_sharpe(window_excess, cov_m, prev_weights, settings)

        old_w = dict(zip(symbols, prev_weights))
        new_w = dict(zip(symbols, w_new))
        drag = tc.get_cost(new_weights=new_w, old_weights=old_w)
        V_after_cost = V * (1.0 - float(drag))

        port_ret = float(np.dot(w_new, r_next))
        V = V_after_cost * (1.0 + port_ret)

        prev_weights = w_new.copy()
        values.append((dates[t], V))

    df_values = pd.DataFrame(values, columns=["date", "AIQ"]).set_index("date")
    ret_m = df_values["AIQ"].pct_change().dropna()
    rf_eval = _load_rf_aligned(ret_m.index)
    if rf_eval is None:
        rf_eval = pd.Series(0.0, index=ret_m.index, name="rf")
    else:
        rf_eval = rf_eval.reindex(ret_m.index).ffill().rename("rf")
    return df_values, ret_m, rf_eval

def _num_logits_for_mode(weight_mode: str) -> int:
    if weight_mode == "free":
        return 3
    if weight_mode == "simplex":
        return 4
    if weight_mode == "basic_eta":
        return 1
    raise ValueError(f"Unknown weight_mode '{weight_mode}'")

def _build_param_space(lookback: int):
    PS = DEFAULTS["param_space"].copy()
    L  = _num_logits_for_mode(DEFAULTS["weight_mode"])
    logits_spec = PS.get("logits", None)
    if (logits_spec is None) or (len(logits_spec) != L):
        PS["logits"] = [(-4.0, 4.0)] * L
    g_rng = PS.get("gamma", None)
    if g_rng is None:
        g_rng = (-DEFAULTS["u_gamma"], DEFAULTS["u_gamma"])
    e_rng = PS.get("epsilon", None)
    if e_rng is None:
        e_rng = (0.0, float(max(1, lookback - 1)))
    thr = PS.get("threshold", 0.05)
    thr_choices = [float(thr)] if isinstance(thr, (int, float)) else list(thr)
    space = dict(
        logits=PS["logits"],
        delta=PS.get("delta", (1.0, 2.0)),
        gamma=g_rng,
        epsilon=e_rng,
        threshold=thr_choices,
        center_method=PS.get("center_method", ["mean","median","zero"]),
        scale_method=PS.get("scale_method", ["vols","pair_max","pair_avg","pair_min"]),
    )
    return space
    
def _sample_params(rng: np.random.Generator, param_bounds: dict, method: str) -> dict:
    L = len(param_bounds['logits'])
    logits = [rng.uniform(*param_bounds['logits'][i]) for i in range(L)]
    delta = rng.uniform(*param_bounds['delta'])
    gamma = rng.uniform(*param_bounds['gamma'])
    epsilon = rng.uniform(*param_bounds['epsilon'])
    threshold = rng.choice(param_bounds['threshold'])
    center_method = rng.choice(param_bounds['center_method'])
    scale_method  = rng.choice(param_bounds['scale_method'])
    return dict(
        iq_method=method,
        logits=logits, delta=delta, gamma=gamma, epsilon=epsilon,
        center_method=center_method, scale_method=scale_method,
        threshold=threshold, PSD_check=DEFAULTS['PSD_check'],
        gamma_mode=DEFAULTS['gamma_mode'], u_gamma=DEFAULTS['u_gamma'], a_floor=DEFAULTS['a_floor'],
    )

def _search_best_params(prcs: pd.DataFrame, method: str, bt_cfg) -> tuple[dict, float]:
    rng = np.random.default_rng(DEFAULTS['seed'])
    space = _build_param_space(bt_cfg.lookback)

    # accept 1, 3, or 4 logits depending on weight_mode
    L = len(space["logits"])
    assert L in (1, 3, 4), "Expected 1 ('basic_eta'), 3 ('free'), or 4 ('simplex') logits."
    print(f"[aiq_mvo] weight_mode={DEFAULTS['weight_mode']} → {L} logits")

    if (not _HAVE_OPTUNA) or int(DEFAULTS['n_trials']) <= 1:
        if DEFAULTS.get("seed_identity_trial", False):
            lo = float(DEFAULTS.get("identity_logit_lo", -4.0))
            hi = float(DEFAULTS.get("identity_logit_hi",  4.0))

            d_lo, d_hi = space["delta"]
            g_lo, g_hi = space["gamma"]
            e_lo, e_hi = space["epsilon"]

            # optuna seed strategy per weight_mode
            wm = DEFAULTS.get("weight_mode", "free")
            if wm == "basic_eta":
                # single logit, eta. Use low logit to bias starting mass to w_T 
                seed_logits = [lo]  # L == 1
            elif wm == "simplex":
                # three lows and one high to bias starting mass to w_I
                seed_logits = [lo, lo, lo, hi]  # L == 4
            else:
                # "free": three independent lows to bias starting mass to w_I
                seed_logits = [lo, lo, lo]  # L == 3

            cov_params = dict(
                iq_method=method,
                logits=seed_logits,
                delta=0.5 * (d_lo + d_hi),
                gamma=0.5 * (g_lo + g_hi),
                epsilon=0.5 * (e_lo + e_hi),
                threshold=space["threshold"][0],
                center_method=space["center_method"][0],
                scale_method=space["scale_method"][0],
                PSD_check=DEFAULTS["PSD_check"],
                gamma_mode=DEFAULTS["gamma_mode"],
                u_gamma=DEFAULTS["u_gamma"],
                a_floor=DEFAULTS["a_floor"],
            )
        else:
            cov_params = _sample_params(rng, space, method)

        _, ret_f, rf_f = backtest_aiq_mvo(prcs, cov_params, bt_cfg)
        return cov_params, float(annualized_sharpe(ret_f, rf_f))
  
    def objective(trial):
        L = len(space["logits"])
        logits = [trial.suggest_float(f"logit_{i}", *space["logits"][i]) for i in range(L)]
        delta  = trial.suggest_float("delta", *space["delta"])
        gamma  = trial.suggest_float("gamma", *space["gamma"])
        epsilon= trial.suggest_float("epsilon", *space["epsilon"])
        threshold     = trial.suggest_categorical("threshold", space["threshold"])
        center_method = trial.suggest_categorical("center_method", space["center_method"])
        scale_method  = trial.suggest_categorical("scale_method", space["scale_method"])

        cov_params = dict(
            iq_method=method,
            logits=logits, delta=delta, gamma=gamma, epsilon=epsilon,
            center_method=center_method, scale_method=scale_method,
            threshold=threshold, PSD_check=DEFAULTS["PSD_check"],
            gamma_mode=DEFAULTS["gamma_mode"], u_gamma=DEFAULTS["u_gamma"], a_floor=DEFAULTS["a_floor"],
        )
        _, ret_m, rf_m = backtest_aiq_mvo(prcs, cov_params, bt_cfg)
        return annualized_sharpe(ret_m, rf_m)

    study = optuna.create_study(direction="maximize")
    study.sampler = optuna.samplers.TPESampler(seed=DEFAULTS['seed'], multivariate=True, group=True)
    study.pruner  = optuna.pruners.MedianPruner(n_startup_trials=int(max(5, 0.2*DEFAULTS['n_trials'])))

    if DEFAULTS.get("seed_identity_trial", False):
        lo = float(DEFAULTS.get("identity_logit_lo", -4.0))
        hi = float(DEFAULTS.get("identity_logit_hi",  4.0))
    
        d_lo, d_hi = space["delta"]; g_lo, g_hi = space["gamma"]; e_lo, e_hi = space["epsilon"]
        mid_delta   = 0.5 * (d_lo + d_hi)
        mid_gamma   = 0.5 * (g_lo + g_hi)
        mid_epsilon = 0.5 * (e_lo + e_hi)
    
        seed_threshold     = space["threshold"][0]
        seed_center_method = space["center_method"][0]
        seed_scale_method  = space["scale_method"][0]
    
        L = len(space["logits"])
        seed = {
            "delta": mid_delta, "gamma": mid_gamma, "epsilon": mid_epsilon,
            "threshold": seed_threshold,
            "center_method": seed_center_method,
            "scale_method": seed_scale_method,
        }
        if DEFAULTS["weight_mode"] == "simplex":
            # [lo, lo, lo, hi]
            for i in range(L):
                seed[f"logit_{i}"] = hi if i == L-1 else lo
        else:  # "free" -> [lo, lo, lo]
            for i in range(L):
                seed[f"logit_{i}"] = lo
    
        study.enqueue_trial(seed)

    study.optimize(objective, n_trials=int(DEFAULTS['n_trials']))

    best = study.best_params
    L = len(space["logits"])
    logits = [best[f"logit_{i}"] for i in range(L)]
    best_json = dict(
        iq_method=method,
        logits=logits,
        delta=best["delta"], gamma=best["gamma"], epsilon=best["epsilon"],
        center_method=best["center_method"], scale_method=best["scale_method"],
        threshold=best["threshold"], PSD_check=DEFAULTS["PSD_check"],
        gamma_mode=DEFAULTS["gamma_mode"], u_gamma=DEFAULTS["u_gamma"], a_floor=DEFAULTS["a_floor"],
        weight_mode=DEFAULTS["weight_mode"],
        psd_repair=DEFAULTS["psd_repair"],
        target_min_eig=DEFAULTS["target_min_eig"],
        target_kappa=DEFAULTS["target_kappa"],
    )

    return best_json, float(study.best_value)

def main():
    prcs = _load_prices_subset(DEFAULTS['prices_csv'], n_assets=DEFAULTS['n_assets'])
    if DEFAULTS['begin_date'] or DEFAULTS['end_date']:
        prcs = prcs.loc[DEFAULTS['begin_date']: DEFAULTS['end_date']]

    bt_cfg = BacktestConfig(lookback=int(DEFAULTS['lookback']), cost_bps=float(DEFAULTS['cost_bps']))

    results = {}
    for method in DEFAULTS['opt_methods']:
        best_params, best_val = _search_best_params(prcs, method, bt_cfg)
        results[method] = dict(best_params=best_params, best_value=best_val)

        out_name = DEFAULTS['export_json'].get(method, f"aiq_params_{method}.json")
        out_path = os.path.join(_script_dir(), out_name)
        with open(out_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"[aiq_mvo] {method} best Sharpe={best_val:.4f}; wrote {out_path}")

    print("\n[aiq_mvo] Summary:")
    for m, r in results.items():
        print(f"  {m}: Sharpe={r['best_value']:.4f} -> {DEFAULTS['export_json'].get(m)}")

if __name__ == "__main__":
    main()
