"""
Name    : diq_mvo.py
Author  : William Smyth & Layla Abu Khalaf
Contact : drwss.academy@gmail.com
Date    : 26/12/2025
Desc    : run MVO for MaxSR with atomic-IQ (AIQ1/AIQ2) alongside shrinkage & RMT.
"""
import os
import pickle
import re
import sys
import numpy as np
import pandas as pd
from diq_mvo_optimizer import calc_assets_moments
from diq_mvo_optimizer import portfolio_optimizer
from diq_mvo_trans_cost import TransCost
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
OOS_DIR  = os.path.join(BASE_DIR, "OOS_results")

try:
    from defaults import DEFAULTS as _D
except Exception:
    _D = None

def detect_label() -> str:
    return "maxsr"

LABEL = detect_label()      
STRATEGY_LABEL = 'maxSharpe'

def _load_csv_series(csv_path, column=None):
    """load a dated series (index=Date) from CSV and return a sorted numeric Series."""
    df  = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    ser = df.iloc[:, 0] if column is None else df[column]
    ser = pd.to_numeric(ser, errors="coerce").dropna()
    ser.index = pd.to_datetime(ser.index)
    return ser.sort_index().rename("rf")

def _asof_align(series, idx, method="ffill"):
    """as-of align a series to the target DatetimeIndex."""
    union = series.index.union(idx).sort_values()
    s     = series.reindex(union)
    if method == "ffill":
        s = s.ffill()
    elif method == "bfill":
        s = s.bfill()
    return s.reindex(idx).rename(series.name)

def _load_rfr_aligned_to_returns(csv_path, returns_index,
                                 frequency      ="M",
                                 column         =None,
                                 align_method   ="ffill",
                                 interpretation ="monthly_effective"):
    """
    read risk-free rate CSV and align to returns index.
    - interpretation='monthly_effective'  -> values are already monthly returns
    - interpretation='annualized_yield'   -> convert annual yield to monthly effective return
    """
    rf = _load_csv_series(csv_path, column)
    if interpretation == "annualized_yield":
        y   = rf / 100.0
        ppy = 12 if str(frequency).upper().startswith("M") else 252
        rf  = (1.0 + y) ** (1.0 / ppy) - 1.0
    rf_aligned = _asof_align(rf, returns_index, method=align_method)
    return rf_aligned.rename("rf")

def _load_rf_aligned(index):
    try:
        rf_path = _D["rf_csv"]
        if not os.path.isabs(rf_path):
            rf_path = os.path.join(BASE_DIR, rf_path)
        return _load_rfr_aligned_to_returns(
            rf_path, index,
            frequency      = _D.get("rf_frequency","M"),
            column         = _D.get("rf_column", None),
            align_method   = _D.get("rf_align_method","ffill"),
            interpretation = _D.get("rf_interpretation","monthly_effective"),
        )
    except Exception as e:
        print(f"[diq_mvo] WARNING: could not load risk-free; proceeding without RF. ({e})")
        return None

def _load_mvo_prices():
    """
    load same master prices CSV used in Part‑1 (from defaults.py) and
    slice to the Part‑2 window: [part2_begin_date : part2_end_date].
    If DEFAULTS['part2_begin_date'] is None, it is derived as (end_date + 1 month).
    """
    if _D is None:
        raise RuntimeError("defaults.DEFAULTS not found; cannot resolve prices path for Part‑2.")

    path = _D["prices_csv"]
    if not os.path.isabs(path):
        path = os.path.join(BASE_DIR, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Master prices CSV not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"], dayfirst=False).set_index("Date")
    df = df.resample("M", label="right").last()

    def _to_month(s):
        return pd.Period(str(s), freq="M")

    p1_end = _to_month(_D["end_date"]) if _D.get("end_date") else None
    p2_begin = _D.get("part2_begin_date")
    p2_end   = _D.get("part2_end_date")

    if p2_begin is None and p1_end is not None:
        p2_begin = (p1_end + 1).strftime("%Y-%m")

    if p2_begin or p2_end:
        df = df.loc[p2_begin:p2_end]

    if df.empty:
        raise ValueError("Price DataFrame is empty after Part‑2 date slicing. Check defaults.py dates.")
    return df, path

def main():
    cash_start        = 100000.0
    lookback_win_size = int(_D["lookback"])
    optimization_cost = 10  # bps
    transaction_cost  = 10  # bps

    tc = TransCost(c=transaction_cost)
    prcs, prcs_path = _load_mvo_prices()
    print(f"[diq_mvo] Using MVO price file: {prcs_path}")

    rets = prcs.pct_change().dropna(axis=0)
    prcs = prcs.iloc[1:]  

    rf_series   = _load_rf_aligned(rets.index)
    rets_excess = rets.sub(rf_series, axis=0) if rf_series is not None else rets

    nT, p   = prcs.shape
    symbols = prcs.columns.to_list()

    obj_function_list = [STRATEGY_LABEL]
    cov_function_list = ["HC",
                         "LS1","LS2","LS3","LS4","LS5",
                         "NLS6","NLS7","NLS8",
                         "CRE","SRE","SRE_opt",
                         "AIQ1","AIQ2",
                         "GS1","GS2","GS3","GS_opt"
                        ]

    savepath = "OOS_results"
    os.makedirs(savepath, exist_ok=True)

    account_dict = {}
    for cov_function in cov_function_list:
        account_dict[cov_function] = {}
        for port_name in obj_function_list:
            account_dict[cov_function][port_name] = []
            account_dict[cov_function][port_name].append(
                {
                    "date"       : prcs.index[lookback_win_size - 1].strftime("%Y-%m-%d"),
                    "weights"    : np.array([0] * p),
                    "shares"     : np.array([0] * p),
                    "values"     : np.array([0] * p),
                    "portReturn" : 0,
                    "transCost"  : 0,
                    "weightDelta": 0,
                    "portValue"  : cash_start
                }
            )
    def get_mean_variance_space(
        returns_df: pd.DataFrame,
        obj_function_list: list,
        cov_function: str = "HC",
        freq: str = "monthly",
        prev_port_weights: dict | None = None,
        simulations: int = 0,
        cost: float | None = None,
    ) -> dict:
        port_opt = portfolio_optimizer(min_weight=0, max_weight=1, cov_function=cov_function, freq=freq)
        port_opt.set_returns(returns_df)
        result_dict = {"port_opt": {}, "asset": {}}
        for obj_fun_str in obj_function_list:
            if prev_port_weights is not None and cost is not None:
                weights = port_opt.optimize(obj_fun_str, prev_weights=prev_port_weights[obj_fun_str]["weights"], cost=cost)
            else:
                weights = port_opt.optimize(obj_fun_str)
            ret, std = port_opt.calc_annualized_portfolio_moments(weights=weights)
            result_dict["port_opt"][obj_fun_str] = {"ret_std": (ret, std), "weights": weights}
        for ticker in returns_df:
            _dat = returns_df[ticker]
            ret, std = calc_assets_moments(_dat, freq=freq)
            result_dict["asset"][ticker] = (ret, std)
        return result_dict

    prev_port_weights_dict = {key: None for key in cov_function_list}

    for t in tqdm(range(lookback_win_size, nT)):
        bgn_date    = rets.index[t - lookback_win_size]
        end_date    = rets.index[t - 1]
        end_date_p1 = rets.index[t]

        sub_rets        = rets.iloc[t - lookback_win_size : t]
        sub_rets_excess = rets_excess.iloc[t - lookback_win_size : t]

        prcs_t   = prcs.iloc[t - 1 : t].values[0]
        prcs_tp1 = prcs.iloc[t : t + 1].values[0]
        rets_tp1 = rets.iloc[t : t + 1].values[0]

        opt_ports_dict = {}
        for cov_function in cov_function_list:
            opt_ports_dict[cov_function] = get_mean_variance_space(
                sub_rets_excess, obj_function_list,
                cov_function,
                prev_port_weights=prev_port_weights_dict[cov_function],
                cost=optimization_cost,
            )
            prev_port_weights_dict[cov_function] = opt_ports_dict[cov_function]["port_opt"]

            for port_name in obj_function_list:
                port_tm1 = account_dict[cov_function][port_name][-1]

                port_t = {
                    "date": end_date_p1.strftime("%Y-%m-%d"),
                    "weights"    : opt_ports_dict[cov_function]["port_opt"][port_name]["weights"],
                    "shares"     : None,
                    "values"     : None,
                    "portReturn" : None,
                    "transCost"  : None,
                    "weightDelta": None,
                    "portValue"  : None,
                }

                port_t["portReturn"] = (port_t["weights"] * rets_tp1).sum()

                old_w = dict(zip(symbols, port_tm1["weights"]))
                new_w = dict(zip(symbols, port_t["weights"]))
                drag  = tc.get_cost(new_weights=new_w, old_weights=old_w)  
                port_t["transCost"] = port_tm1["portValue"] * float(drag)

                V_after_cost     = port_tm1["portValue"] - port_t["transCost"]
                port_t["shares"] = V_after_cost * port_t["weights"] / prcs_t
                port_t["values"] = V_after_cost * port_t["weights"]
                port_t["weightDelta"] = np.sum(np.abs(port_t["weights"] - port_tm1["weights"]))

                port_t["portValue"] = V_after_cost * (1 + port_t["portReturn"])
                account_dict[cov_function][port_name].append(port_t)

    with open(f"{savepath}/result.pickle", "wb") as f:
        pickle.dump(account_dict, f)

    for cov_func in cov_function_list:
        portAccountDF = pd.DataFrame.from_dict(
            {
                (port_name, account["date"]): {
                    "value"   : account["portValue"],
                    "return"  : account["portReturn"],
                    "trans"   : account["transCost"],
                    "turnover": account["weightDelta"]
                }
                for port_name in account_dict[cov_func].keys()
                for account in account_dict[cov_func][port_name]
            },
            orient="index",
        )

        portAccountDF.reset_index(inplace=True)
        portAccountDF.columns = ["port", "date", "value", "return", "trans", "turnover"]
        portAccountDF.pivot(index="date", columns="port", values="value").to_csv(
            f"{savepath}/{cov_func}_value.csv"
        )
        portAccountDF.pivot(index="date", columns="port", values="return").to_csv(
            f"{savepath}/{cov_func}_return.csv"
        )
        portAccountDF.pivot(index="date", columns="port", values="trans").to_csv(
            f"{savepath}/{cov_func}_trans.csv"
        )
        portAccountDF.pivot(index="date", columns="port", values="turnover").to_csv(
            f"{savepath}/{cov_func}_turnover.csv"
        )

    print(f"Detected objective: {LABEL} -> strategy '{STRATEGY_LABEL}'.")
    print("Run complete. Outputs saved under:", savepath)

if __name__ == '__main__':
    main()
